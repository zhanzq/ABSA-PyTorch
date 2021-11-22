#!/usr/bin/env bash
python train.py \
    --model_name bert_spc \
    --dataset xp \
    --pretrained_bert_name /Users/zhanzq/Downloads/models/bert-base-chinese \
	--best_model_path state_dict/bert_spc_xp.bin
