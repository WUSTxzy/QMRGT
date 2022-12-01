#!/bin/bash

TASK=15
MODEL=vl-bert
MODEL_CONFIG=vl-bert_base
TASKS_CONFIG=vl-bert_test_tasks
PRETRAINED=checkpoints/${MODEL}/GQA_${MODEL_CONFIG}/pytorch_model_1.bin
OUTPUT_DIR=results/gqa/${MODEL}

cd ../../..
python eval_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_1.bin-/test_result.json \
  --truth_file ${ANNOS}/testdev_balanced_questions.json
