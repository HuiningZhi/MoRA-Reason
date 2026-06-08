#!/bin/bash

MODEL_PATH="/root/autodl-tmp/output/tiny-llava-Phi-4-mini-instruct-siglip2-so400m-patch14-384-base-finetune-down-moelora2"
MODEL_NAME="exp23"
TASK_NAME="test_rad"  #test_rad,test_pvqa,test_slake
EVAL_DIR="eval_dir"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --model-base $MODEL_NAME \
    --question-file "/root/autodl-tmp/dataset/3vqa/$TASK_NAME.jsonl" \
    --image-folder "/root/autodl-tmp/dataset/3vqa/images" \
    --answers-file $EVAL_DIR/$TASK_NAME/answers/$MODEL_NAME.jsonl \
    --temperature 0.4 \
    --conv-mode phi-3

# python -m tinyllava.eval.run_eval \
#     --gt /root/autodl-tmp/dataset/3vqa/$TASK_NAME.json \
#     --candidate "" \
#     --pred $EVAL_DIR/$TASK_NAME/answers/$MODEL_NAME.jsonl

python -m tinyllava.eval.run_eval2 \
    --gt /root/autodl-tmp/dataset/3vqa/$TASK_NAME.json \
    --pred $EVAL_DIR/$TASK_NAME/answers/$MODEL_NAME.jsonl