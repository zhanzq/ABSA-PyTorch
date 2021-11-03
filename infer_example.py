# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/3
#

import torch
import torch.nn.functional as F

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, get_example

from models.aen import AEN_BERT
from models.bert_spc import BERT_SPC
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT

from transformers import BertModel


class Inference:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            self.tokenizer = build_tokenizer(
                data_files=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                tokenizer_file='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                embedding_file='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} from {1}'.format(opt.model_name, opt.state_dict_path))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, aspect):
        data = get_example(utt=text, aspect=aspect, polar=None, tokenizer=self.tokenizer)
        inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols]
        outputs = self.model(inputs)
        probs = F.softmax(outputs, dim=-1).cpu().numpy()[0]

        return probs


class Option(object):
    def __init__(self, ):
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

        self.dataset = 'xp'
        self.model_name = 'bert_spc'
        self.inputs_cols = input_cols[self.model_name]
        self.model_class = model_classes[self.model_name]
        self.dataset_file = dataset_files[self.dataset]
        # set your trained models here
        self.bert_dim = 768
        self.embed_dim = 300
        self.hidden_dim = 300
        self.max_seq_len = 40
        if torch.cuda.is_available():   # run on gpu of server
            self.state_dict_path = 'state_dict/bert_spc_xp.bin'
            self.pretrained_bert_name = '/data/zhanzhiqiang/models/bert-base-chinese'
            # self.pretrained_bert_name = "/opt/nas/xp-absa-boot/0.0.1/"  # for deployment on dev/test/pre environment
            # self.state_dict_path = '/opt/nas/xp-absa-boot/0.0.1/bert_spc_xp.bin'
        else:   # run on cpu locally
            self.state_dict_path = 'state_dict/bert_spc_xp.bin'
            self.pretrained_bert_name = '/Users/zhanzq/Downloads/models/bert-base-chinese'

        self.SRD = 3
        self.hops = 3
        self.dropout = 0
        self.polarities_dim = 3
        self.local_context_focus = 'cdm'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    opt = Option()
    # set your trained models here
    # opt.state_dict_path = 'state_dict/ian_restaurant_acc0.7911'
    # opt.state_dict_path = 'state_dict/bert_spc_xp_val_acc_0.9319'

    inf = Inference(opt)
    prompt = "请输入用户语句:"
    print(prompt)
    test_sentence = input()
    while test_sentence:
        prompt = "请输入aspect,可为空:"
        print(prompt)
        aspect = input()
        if not aspect:
            test_sentence += "[UNK]"
            aspect = "[UNK]"
        t_probs = inf.evaluate(test_sentence, aspect)
        print(t_probs.argmax(axis=-1) - 1)

        prompt = "请输入用户语句:"
        print(prompt)
        test_sentence = input()


if __name__ == '__main__':
    main()

