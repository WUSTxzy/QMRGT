#!/bin/bash

TASK=15
ANNOS=data/gqa/annotations
MODEL=vilbert
MODEL_CONFIG=vilbert_base
TASKS_CONFIG=vilbert_test_tasks
PRETRAINED=checkpoints/gqa/vilbert_test_final1/GQA_vilbert_base/pytorch_model_19.bin
OUTPUT_DIR=results/gqa/vilbert_gnn_final1


cd ../../..
python eval_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_19.bin-/test_result.json \
  --truth_file ${ANNOS}/testdev_all_questions.json


