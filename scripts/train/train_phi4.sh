DATA_PATH=/root/autodl-tmp/dataset/PubMedVision_Alignment_VQA.json #pretrain annotation file path
IMAGE_PATH=/root/autodl-tmp/dataset #pretrain image dir

FINETUNE_DATA_PATH=/root/autodl-tmp/dataset/PubMedVision_InstructionTuning_VQA.json #finetune annotation file path
FINETUNE_IMAGE_PATH=/root/autodl-tmp/dataset #finetune image dir

FINETUNE_DOWN_DATA_PATH=/root/autodl-tmp/dataset/3vqa/train_all.json #finetune downstream annotation file path
FINETUNE_DOWN_IMAGE_PATH=/root/autodl-tmp/dataset/3vqa/images #finetune downstream image dir

REASON_DATA_PATH=/root/autodl-tmp/dataset/3vqa/train_allreason2.json
REASON_IMAGE_PATH=/root/autodl-tmp/dataset/3vqa/images

LLM_VERSION=/root/autodl-tmp/model/Phi-4-mini-instruct # llm path in huggingface
VT_VERSION=/root/autodl-tmp/model/siglip2-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi-4 #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm


# bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" 
# bash scripts/train/share/pretrain_share.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" 
# bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/finetune_down.sh "$FINETUNE_DOWN_DATA_PATH" "$FINETUNE_DOWN_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"

# bash scripts/train/finetune_down_reason.sh "$REASON_DATA_PATH" "$REASON_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/finetune_down_moelora.sh "$FINETUNE_DOWN_DATA_PATH" "$FINETUNE_DOWN_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
bash scripts/train/finetune_down_moelora.sh "$REASON_DATA_PATH" "$REASON_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
