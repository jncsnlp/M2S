#!/bin/bash
#./playground/data/llava_v1_5_mix665k.json \
#for up in 1 2 3 4 5 6 7 8 9 10 11 12 13 14
#do
deepspeed /home/jncsnlp/lxf/LLaVA/llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /home/jncsnlp/lxf/LLaVA/scripts/zero3.json \
    --model_name_or_path /home/jncsnlp/lxf/llava-v1.5-7b \
    --version v1 \
    --data_path /home/jncsnlp/lxf/data/mvsa-s/few-shot1-1.json \
    --image_folder /home/jncsnlp/lxf/dataset/MVSA_Single/data \
    --vision_tower /home/jncsnlp/lxf/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task-lora-1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
        #--dev_path "/home/jncsnlp/lxf/dev.json""/home/jncsnlp/lxf/data/t2015/dev.json"--dev_path "/home/jncsnlp/lxf/data/t2015/dev-prompt5.json" \
#done