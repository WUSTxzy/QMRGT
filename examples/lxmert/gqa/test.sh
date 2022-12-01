#!/bin/bash

TASK=15
ANNOS=data/gqa/annotations
MODEL=lxmert
MODEL_CONFIG=lxmert
TASKS_CONFIG=lxmert_test_tasks
PRETRAINED=checkpoints/gqa/lxmert_gnn/GQA_lxmert/pytorch_model_best.bin
OUTPUT_DIR=results/gqa/lxmert_gnn

cd ../../..
python eval_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_best.bin-/test_result.json \
  --truth_file ${ANNOS}/testdev_balanced_questions.json
