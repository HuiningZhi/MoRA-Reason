DATA_PATH=LLaVA_data_all/alignment_500k_filter.json
SHARE_PRETRAIN_DATA_PATH=LLaVA_data_all/alignment_500k_filter.json
SHARE_FINETUNE_DATA_PATH=LLaVA_data_all/instruct_60k_filter.json
IMAGE_PATH=LLaVA_data_all/all_images
SHARE_PRETRAIN_IMAGE_PATH=LLaVA_data_all/all_images
SHARE_FINETUNE_IMAGE_PATH=LLaVA_data_all/all_images

LLM_VERSION=models/microsoft/phi-2
VT_VERSION=models/openai/clip-vit-large-patch14-336
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=phi
VERSION=share
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=3072



bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/share/pretrain_share.sh "$SHARE_PRETRAIN_DATA_PATH" "$SHARE_PRETRAIN_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" 
bash scripts/train/share/finetune_share.sh "$SHARE_FINETUNE_DATA_PATH" "$SHARE_FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
