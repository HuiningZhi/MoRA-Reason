DATA_PATH=data/3vqa/test_rad.json #pretrain annotation file path
FINETUNE_DATA_PATH=data/3vqa/test_rad.json #finetune annotation file path
IMAGE_PATH=data/3vqa/images #pretrain image dir
FINETUNE_IMAGE_PATH=data/3vqa/images #finetune image dir

LLM_VERSION=models/microsoft/phi-2 # llm path in huggingface
VT_VERSION=models/google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=phi
VERSION=base-lora-zero2-r128
PRETRAIN_TRAIN_RECIPE=common
FINETUNE_TRAIN_RECIPE=lora
MODEL_MAX_LENGTH=3072


bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$PRETRAIN_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/lora/finetune_lora.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$FINETUNE_TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
