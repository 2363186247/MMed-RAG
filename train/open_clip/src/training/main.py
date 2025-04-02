import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync

import debugpy

# 假设你使用的是环境变量 RANK 来标识每个进程
# rank = int(os.getenv('RANK', '0'))
# port = 5678 + rank  # 基础端口 + 进程ID

# debugpy.listen(port)
# print(f"Process {rank} waiting for debugger to attach on port {port}...")
# debugpy.wait_for_client()
LATEST_CHECKPOINT_NAME = "epoch_latest.pt"

# 设置随机种子以确保实验的可复现性
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        # 将模型名称中的斜杠 / 替换为短横线 -，以确保生成的名称可以安全地用于文件系统路径或 URI。
        # 例如，模型名称 clip/vit-b32 会被转换为 clip-vit-b32。

        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # 使用 datetime.now() 获取当前时间，并格式化为字符串，格式为 年_月_日-时_分_秒。
        # 例如，2025-04-02 15:30:45 会被格式化为 2025_04_02-15_30_45。

        if args.distributed:
            # sync date_str from master to all ranks
            # 将主节点的时间字符串同步到所有节点。
            # 确保所有节点使用相同的时间字符串，避免生成不同的实验名称。
            date_str = broadcast_object(args, date_str)
        
        # 动态生成实验名称
        # 例如：2025_04_02-15_30_45-model_clip-vit-b32-lr_0.0005-b_64-j_8-p_fp16
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # 检查是否恢复最新的实验，即命令行参数 --resume 是否设置为 'latest'
    # 如果设置为 'latest'，则表示用户希望恢复最新的实验检查点。
    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):   # 检查是否为主节点
        os.makedirs(log_base_path, exist_ok=True)   # 创建日志目录
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'  # 设置日志文件名
        args.log_path = os.path.join(log_base_path, log_filename)   # 设置完整的日志文件路径

        # 如果日志文件已经存在，并且用户没有指定恢复最新的实验（resume_latest 为 False），则认为实验已经存在。
        # 打印错误信息，提示用户使用 --name 参数指定一个新的实验名称。
        # 返回 -1，终止程序。
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    # 这行代码根据命令行参数 --debug 的值设置日志级别：
    # 如果 args.debug 为 True，则日志级别为 logging.DEBUG，表示记录所有调试信息。
    # 如果 args.debug 为 False，则日志级别为 logging.INFO，表示记录一般信息和更高级别的日志（如警告和错误）。
    #   *日志级别控制了哪些日志消息会被记录：
    #       -DEBUG：记录所有消息，包括调试信息。
    #       -INFO：记录一般信息、警告和错误。
    #       -WARNING：记录警告和错误。
    #       -ERROR：仅记录错误。
    #       -CRITICAL：仅记录严重错误。
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    # Weights & Biases：wandb 是一个实验跟踪工具，用于记录实验的超参数、指标和模型。
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    # tensorboard 是一个可视化工具，用于监控训练过程中的指标（如损失、准确率等）。
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    # 设置 checkpoint 存储路径，存储在实验日志目录下的 checkpoints 文件夹中。
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    # 设置 tensorboard 存储路径，存储在实验日志目录下的 tensorboard 文件夹中。
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None  # 存储恢复 checkpoint 的路径
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            # 检查不兼容的选项
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            # 通过主节点找到最新的检查点路径，并将其同步到所有节点。
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    # 处理远程同步逻辑。
    # 如果用户通过命令行参数 --remote-sync 指定了远程同步路径，
    # 则会在主节点上执行远程同步操作，
    # 并启动一个后台进程定期同步本地和远程的实验数据。
    remote_sync_process = None  # 存储远程同步的后台进程对象
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(   # 初次远程同步
            os.path.join(args.logs, args.name),         # 实验日志目录
            os.path.join(args.remote_sync, args.name),  # 远程同步目录
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(   # 如果初次同步成功，则启动一个后台进程，定期执行远程同步操作。
            args.remote_sync_frequency,                 # 同步频率
            os.path.join(args.logs, args.name),         # 实验日志目录
            os.path.join(args.remote_sync, args.name),  # 远程同步目录
            args.remote_sync_protocol
        )
        remote_sync_process.start() # 启动后台进程

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    # Horovod 是一种分布式深度学习框架，支持多节点多 GPU 训练
    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    # 分布式模式:使用 PyTorch 的分布式训练功能
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        # 当前代码不支持梯度累积（accum_freq > 1）下的蒸馏模式。
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        # 当前代码不支持使用 coca 模型进行蒸馏。
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        # 如果是一个长度为 1 的列表或元组（例如 [224]），则将其转换为整数（224）
        # 这样可以简化后续代码中对图像尺寸的处理。
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)   # 初始的对数尺度, different from CLIP
        model_kwargs['init_logit_bias'] = -10           # 初始的对数偏置
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )


    # for name, param in model.named_parameters():#TODO model trainable weights
        
    #     param.requires_grad = False

    #     # 检查参数名称，决定是否为顶层参数
    #     # 这里假设顶层为文本编码器的第11层和图像编码器的第10层和第11层
    #     # if "text.transformer.encoder.layer.11" in name or \
    #     #     "text.proj" in name or \
    #     #     "visual.trunk.blocks.11" in name or \
    #     #     "visual.trunk.norm" in name or \
    #     #     "visual.head.proj" in name:
    #     #     param.requires_grad = True

    #     if "text.proj" in name or \
    #         "visual.trunk.norm" in name or \
    #         "visual.head.proj" in name:
    #         param.requires_grad = True

    # 打印模型参数的可训练状态
    # 检查模型中哪些参数是可训练的，哪些参数被冻结
    for name, param in model.named_parameters():
        print(f"{name}: trainable={param.requires_grad}")
    
    # 蒸馏
    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
        )
    
    # 使用 bitsandbytes 提供的高效线性层替换模型中的标准线性层
    #  bitsandbytes 的线性层可以显著减少模型的内存占用，同时保持较高的计算效率
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    # 如果用户通过 --trace 参数启用了模型的 TorchScript 编译，则调用 trace_model 函数对模型进行编译。
    # TorchScript 是 PyTorch 提供的一种模型优化工具，可以将模型转换为静态图，从而提高推理速度。
    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    # 冻结模型的图像编码器
    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,    # 指定图像编码器中未冻结的组
            freeze_bn_stats=args.lock_image_freeze_bn_stats)    # 是否冻结批归一化（BatchNorm）
        
    # 冻结模型的文本编码器
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,     # 指定文本编码器中未冻结的层
            freeze_layer_norm=args.lock_text_freeze_layer_norm) # 是否冻结层归一化（LayerNorm）

    # 启用梯度检查点，梯度检查点是一种节省显存的技术
    # 通过在前向传播中丢弃部分中间结果，减少显存占用，但会增加反向传播的计算开销。
    if args.grad_checkpointing: 
        model.set_grad_checkpointing()

    # 记录模型和参数信息
    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # 分布式训练支持
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
        # 将模型中的批归一化层转换为同步批归一化（SyncBatchNorm），以在多 GPU 上同步统计信息。
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
        # 启用静态图优化（仅在 PyTorch 新版本中支持）
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        # 确保模型未被 TorchScript 编译，因为编译后的模型不支持训练。
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        # 遍历模型的所有参数，并根据 exclude 和 include 函数将参数分为两组：
        # -gain_or_bias_params：
        #       包含偏置、批归一化参数等特殊参数。
        #       这些参数通常不需要权重衰减（weight_decay=0）。
        # -rest_params：
        #       包含普通参数（如权重矩阵）。
        #       这些参数通常需要权重衰减（weight_decay=args.wd）。
        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(            # AdamW 优化器
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,                     # 学习率
            betas=(args.beta1, args.beta2), # 动量参数
            eps=args.eps,                   # 数值稳定性参数
        )
        
        # Horovod 分布式优化器
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)   # 广播模型参数
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)       # 广播优化器状态

        # 如果启用了混合精度训练（args.precision == "amp"），创建 GradScaler 对象。
        # GradScaler 用于动态调整梯度缩放因子，避免数值不稳定问题。
        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            
            # 如果当前不是分布式训练模式，但检查点中的参数名称以 module. 开头（通常是分布式训练保存的模型状态），则移除 module. 前缀。
            # 这是为了兼容分布式和非分布式训练的模型状态。
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            
            # 加载模型状态
            model.load_state_dict(sd)

            # 恢复优化器状态
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            
            # 恢复混合精度缩放器状态
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            
            # 记录恢复信息（包括 checkpoint 路径和恢复的 epoch ）
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            # 如果检查点中不包含 epoch 字段，说明这是一个仅包含模型状态的检查点（例如用于微调或评估）。
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    # 确保至少指定了一个数据集
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None    # 学习率调度器
    if 'train' in data and optimizer is not None:
        # 计算总的训练步数
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":           # 余弦退火学习率调度器
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":          # 常数学习率调度器
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown": # 常数冷却学习率调度器
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                # cooldown_steps：冷却阶段的步数。
                # args.lr_cooldown_power：冷却阶段的学习率衰减幂。
                # args.lr_cooldown_end：冷却阶段的最终学习率。
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    
    # 初始化 TensorBoard
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    # 初始化 Weights & Biases (WandB)
    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples    # 训练数据集的样本数量
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples    # 验证数据集的样本数量
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')   # 监控模型的参数和梯度
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model

    # 支持 PyTorch 2.0 的模型编译
    if args.torchcompile:
        logging.info('Compiling model...')
        # 将模型转换为静态图，从而优化推理和训练性能
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        # 如果启用了 bitsandbytes 的低精度线性层（args.use_bnb_linear），则将模型转换为推理模式。
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        # 训练一个 epoch
        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        # 如果数据加载器中包含验证数据（如 'val'、'imagenet-val' 或 'imagenet-v2'），调用 evaluate 函数对模型进行评估。
        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        # 如果启用了日志保存（args.save_logs），并且当前 epoch 是每 10 轮的最后一轮（epoch % 10 == 9），则保存检查点
        if args.save_logs and epoch%10==9:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            # 如果当前 epoch 是最后一个 epoch，或者满足保存频率（args.save_frequency）的条件，则将检查点保存到文件。
            # 文件名格式为 epoch_{completed_epoch}.pt。
            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            # 删除旧的检查点
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)
            # 保存最新检查点
            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)
    
    # 结束 WandB 记录
    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate() # remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")  # 用于存储复制的代码库
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)  # 获取当前脚本文件的绝对路径
    
    # 向上遍历三层目录，定位到代码库的根目录。
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    # 复制当前代码库到新的目录
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
