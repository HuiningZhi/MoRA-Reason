#!/bin/bash

MODEL_PATH="output/Final/tiny-llava-Phi-3.5-mini-instruct-siglip-so400m-patch14-384-base-finetune-share"
MODEL_NAME="exp23"
TASK_NAME="test_pvqa"
EVAL_DIR="eval_dir"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --model-base $MODEL_NAME \
    --question-file dataset/data/3vqa/$TASK_NAME.jsonl \
    --image-folder dataset/data/3vqa/images \
    --answers-file $EVAL_DIR/$TASK_NAME/answers/$MODEL_NAME.jsonl \
    --temperature 0.4 \
    --conv-mode phi-3

# python -m tinyllava.eval.run_eval \
#     --gt LLaVA_data_all/3vqa/$TASK_NAME.json \
#     --candidate "" \
#     --pred $EVAL_DIR/$TASK_NAME/answers/$MODEL_NAME.jsonl

python -m tinyllava.eval.run_eval2 \
    --gt dataset/data/3vqa/$TASK_NAME.json \
    --pred $EVAL_DIR/$TASK_NAME/answers/$MODEL_NAME.jsonl