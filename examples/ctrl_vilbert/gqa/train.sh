#!/bin/bash

TASK=13
MODEL=ctrl_vilbert
MODEL_CONFIG=ctrl_vilbert_base
TASKS_CONFIG=ctrl_trainval_tasks
PRETRAINED=checkpoints/conceptual_captions/${MODEL}/${MODEL_CONFIG}/pytorch_model_9.bin
OUTPUT_DIR=checkpoints/gqa/ctrl_vilbert
LOGGING_DIR=logs/gqa_ctrl_vilbert


cd ../../..
python train_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--adam_epsilon 1e-6 --adam_betas 0.9 0.999 --weight_decay 0.01 --warmup_proportion 0.1 --clip_grad_norm 0.0 \
	--output_dir ${OUTPUT_DIR} \
	--logdir ${LOGGING_DIR} \
#	--resume_file ${OUTPUT_DIR}/GQA_${MODEL_CONFIG}/pytorch_ckpt_latest.tar
# --resume_file checkpoints/gqa/vilbert_test/GQA_vilbert_base/pytorch_ckpt_latest.tar \
