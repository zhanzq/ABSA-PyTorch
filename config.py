# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/22
#

import os
import numpy
import torch
import random
import argparse
from models import AEN_BERT, BERT_SPC, LCF_BERT
from models import LSTM, TC_LSTM, TD_LSTM, ATAE_LSTM
from models import IAN, MemNet, RAM, Cabasc, TNet_LF, AOA, MGAN, ASGCN

model_classes = {
    'aoa': AOA,
    'ian': IAN,
    'ram': RAM,
    'mgan': MGAN,
    'asgcn': ASGCN,
    'memnet': MemNet,
    'cabasc': Cabasc,
    'tnet_lf': TNet_LF,
    'bert_spc': BERT_SPC,
    'aen_bert': AEN_BERT,
    'lcf_bert': LCF_BERT,
    'lstm': LSTM,
    'td_lstm': TD_LSTM,
    'tc_lstm': TC_LSTM,
    'atae_lstm': ATAE_LSTM,
}

dataset_files = {
    'xp': {
        'train': './datasets/xp/train.txt',
        'test': './datasets/xp/train.txt',
    },
    'twitter': {
        'train': './datasets/acl-14-short-data/train.raw',
        'test': './datasets/acl-14-short-data/test.raw'
    },
    'restaurant': {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    },
    'laptop': {
        'train': './datasets/semeval14/Laptops_Train.xml.seg',
        'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
    },
}

input_cols = {
    'lstm': ['text_indices'],
    'aoa': ['text_indices', 'aspect_indices'],
    'ian': ['text_indices', 'aspect_indices'],
    'atae_lstm': ['text_indices', 'aspect_indices'],
    'memnet': ['context_indices', 'aspect_indices'],
    'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
    'ram': ['text_indices', 'aspect_indices', 'left_indices'],
    'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
    'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
    'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_in_text'],
    'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
    'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
    'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
}

initializers = {
    "xavier_uniform_": torch.nn.init.xavier_uniform_,
    "xavier_normal_": torch.nn.init.xavier_normal_,
    "orthogonal_": torch.nn.init.orthogonal_,
}

optimizers = {
    "sgd": torch.optim.SGD,
    "asgd": torch.optim.ASGD,  # default lr=0.01
    "adam": torch.optim.Adam,  # default lr=0.001
    "adamax": torch.optim.Adamax,  # default lr=0.002
    "rmsprop": torch.optim.RMSprop,  # default lr=0.01
    "adagrad": torch.optim.Adagrad,  # default lr=0.01
    "adadelta": torch.optim.Adadelta,  # default lr=1.0
}

# Hyper Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="bert_spc", type=str)
parser.add_argument("--dataset", default="xp", type=str, help="xp, twitter, restaurant, laptop")
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--initializer", default="xavier_uniform_", type=str)
parser.add_argument("--lr", default=2e-5, type=float, help="try 5e-5, 2e-5 for BERT, 1e-3 for others")
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--l2reg", default=0.01, type=float)
parser.add_argument("--num_epoch", default=5, type=int, help="try larger number (>20) for non-BERT models")
parser.add_argument("--batch_size", default=32, type=int, help="try 16, 32, 64 for BERT models")
parser.add_argument("--eval_batch_size", default=64, type=int, help="use 32, 64 ,128 or larger")
parser.add_argument("--log_step", default=10, type=int)
parser.add_argument("--embed_dim", default=300, type=int)
parser.add_argument("--hidden_dim", default=300, type=int)
parser.add_argument("--bert_dim", default=768, type=int)
parser.add_argument("--pretrained_bert_name", default="bert-base-uncased", type=str)
parser.add_argument("--best_model_path", default="./state_dict/pytorch_model.bin", type=str)
parser.add_argument("--max_seq_len", default=40, type=int)
parser.add_argument("--polarities_dim", default=3, type=int)
parser.add_argument("--hops", default=3, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--device", default=None, type=str, help="e.g. cuda:0")
parser.add_argument("--data_dir", default="datasets/xp/", type=str, help="xp dataset")
parser.add_argument("--seed", default=1234, type=int, help="set seed for reproducibility")
parser.add_argument("--do_train", default=False, type=bool, help="to train the model")
parser.add_argument("--do_eval", default=True, type=bool, help="to evaluate the model")
parser.add_argument("--do_test", default=False, type=bool, help="to test the model")
parser.add_argument("--valid_dataset_ratio", default=0.1, type=float,
                    help="set ratio between 0 and 1 for validation support")

# The following parameters are only valid for the lcf-bert model
parser.add_argument("--local_context_focus", default="cdm", type=str, help="local context focus mode, cdw or cdm")
parser.add_argument("--SRD", default=3, type=int,
                    help="semantic-relative-distance, see the paper of LCF-BERT model")

option = parser.parse_args()
option.optimizer = optimizers[option.optimizer]
option.inputs_cols = input_cols[option.model_name]
option.model_class = model_classes[option.model_name]
option.dataset_file = dataset_files[option.dataset]
option.initializer = initializers[str(option.initializer)]
if option.device is None:
    if torch.cuda.is_available():
        option.device = torch.device("cuda")
    else:
        option.device = torch.device("cpu")
else:
    option.device = torch.device(str(option.device))

seed = option.seed
if seed is not None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
