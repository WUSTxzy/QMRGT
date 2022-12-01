#!/bin/bash
PATH=/home/xzy/myprojects/volta/data/gqa \
python -u convert_vg_gqa_lmdb.py \
--indir ${PATH}/imgfeats  \
--outdir ${PATH}/imgfeats/volta


