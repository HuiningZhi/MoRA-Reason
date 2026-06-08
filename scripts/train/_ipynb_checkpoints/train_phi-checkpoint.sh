DATA_PATH=LLaVA_data_all/alignment_500k_filter.json #pretrain annotation file path
FINETUNE_DATA_PATH=LLaVA_data_all/instruct_60k_filter.json #finetune annotation file path
IMAGE_PATH=LLaVA_data_all/all_images #pretrain image dir
FINETUNE_IMAGE_PATH=LLaVA_data_all/all_images #finetune image dir

LLM_VERSION=models/microsoft/phi-2 # llm path in huggingface
VT_VERSION=models/google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=phi #chat template, other options are: phi, llama, gemmma, etc
VERSION=base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=3072 #max model length for llm


bash scripts/train/pretrain.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
# bash scripts/train/finetune.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
