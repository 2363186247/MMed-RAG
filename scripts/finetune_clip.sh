export CUDA_VISIBLE_DEVICES="0,1"


cd ./train/open_clip/src

# harvard dataset
torchrun --nproc_per_node=2 \
    --master_port=12347 \
    -m training.main \
    --model hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --train-data '/path/to/train.json' \
    --dataset-type radiology \
    --img_root /path/to/img_root \
    --batch-size 512 \
    --precision amp \
    --workers 4 \
    --lr 0.0001 \
    --epochs 360 \
    --val-data "/path/to/val.json" \
    --val-frequency 10 \
    --report-to tensorboard \
    --logs /path/to/checkpoints_saving


