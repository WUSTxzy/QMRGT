#!/bin/bash

TASK=13
ANNOS=data/gqa/annotations
MODEL=ctrl_uniter
MODEL_CONFIG=ctrl_uniter_base
TASKS_CONFIG=ctrl_test_tasks
PRETRAINED=checkpoints/gqa/ctrl_uniter_gnn/GQA_ctrl_uniter_base/pytorch_model_19.bin
OUTPUT_DIR=results/gqa/ctrl_uniter_gnn

cd ../../..
python eval_task.py \
	--config_file config/${MODEL_CONFIG}.json --from_pretrained ${PRETRAINED} \
	--tasks_config_file config_tasks/${TASKS_CONFIG}.yml --task $TASK \
	--output_dir ${OUTPUT_DIR}

python scripts/GQA_score.py \
  --preds_file ${OUTPUT_DIR}/pytorch_model_19.bin-/test_result.json \
  --truth_file ${ANNOS}/testdev_balanced_questions.json
