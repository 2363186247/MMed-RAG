cd ./train/open_clip/src || exit

CUDA_VISIBLE_DEVICES=0,1 python ./retrieve_clip_report.py \
    --img_root /ai/xxr/Datasets/Med/Harvard-FairVLMed \
    --train_json /ai/xxr/LLM/MMed-RAG/data/training/retriever/ophthalmology/harvard_train_7000.json \
    --eval_json /ai/xxr/LLM/MMed-RAG/data/training/retriever/ophthalmology/harvard_val_1000.json \
    --model_name_or_path hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --checkpoint_path /ai/xxr/LLM/MMed-RAG/output/finetune_clip/2025_03_25-05_22_36-model_hf-hub:thaottn-OpenCLIP-resnet50-CC12M-lr_0.0001-b_512-j_4-p_amp/checkpoints/epoch_360.pt \
    --output_path /ai/xxr/LLM/MMed-RAG/output/retrieve_clip_report/harvard \
    --eval_type "test" \
    --fixed_k 5 \
    # --clip_threshold 1.5 \

