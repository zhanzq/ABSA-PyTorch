#!/usr/bin/env bash
python evaluate.py \
    --arch_name bert_spc \
    --data_dir datasets/xp/data_test \
	--do_test \
    --pretrained_bert_name /Users/zhanzq/Downloads/models/bert-base-chinese \
	--best_model_path ~/Downloads/xp_absa_pytorch_model.bin
