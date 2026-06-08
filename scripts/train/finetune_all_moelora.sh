#!/bin/bash
if [ $# -ne 10 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <CONV_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH>"
    exit 1
fi

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
CONV_VERSION="$7"
VERSION="$8"
TRAIN_RECIPE="$9"
MODEL_MAX_LENGTH="${10}"

VT_VARIANT="${VT_VERSION##*/}"
LLM_VARIANT="${LLM_VERSION##*/}"

# ✅ MMOELoRA 配置参数
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1
EXPERT_NUM=4
TASK_NUM=2
TASK_EMBEDDING_DIM=32

# ✅ 使用 DeepSpeed 进行 4 卡训练
deepspeed --include localhost:0 --master_port 29501 tinyllava/train/custom_finetune_moelora.py \
    --deepspeed ./scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --fp16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector frozen \
    --group_by_modality_length False \
    --pretrained_model_path /root/autodl-tmp/output/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune-down-moelora2 \
    --output_dir /root/autodl-tmp/output/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune-down-moelorall \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --expert_num $EXPERT_NUM \
    --task_num $TASK_NUM \
    --task_embedding_dim $TASK_EMBEDDING_DIM \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune_down_moelora