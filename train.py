# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/3
#

import os
import sys
import math
from typing import Tuple

import numpy
import random
import logging
import argparse

from sklearn import metrics
from sklearn.metrics import confusion_matrix

from transformers import BertModel
from time import strftime, localtime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import load_data
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

from models.aen import AEN_BERT
from models.bert_spc import BERT_SPC
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if "bert" in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                data_files=[opt.dataset_file["train"], opt.dataset_file["test"]],
                max_seq_len=opt.max_seq_len,
                tokenizer_file="{0}_tokenizer.dat".format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                embedding_file="{0}_{1}_embedding_matrix.dat".format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        data_dir = opt.data_dir
        train_data, valid_data, test_data = load_data(data_dir=data_dir, with_punctuation=False, tokenizer=tokenizer)

        self.train_dataset = ABSADataset(train_data)
        self.valid_dataset = ABSADataset(valid_data)
        self.test_dataset = ABSADataset(test_data)

        if opt.device.type == "cuda":
            logger.info("cuda memory allocated: {}".format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_untrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_untrainable_params += n_params
        logger.info("> n_trainable_params: %d, n_untrainable_params: %d" % (n_trainable_params, n_untrainable_params))
        logger.info("> training arguments:")
        for arg in vars(self.opt):
            logger.info(">>> {0}: {1}".format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdev = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdev, b=stdev)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        path = None
        max_val_f1 = 0
        max_val_acc = 0
        global_step = 0
        max_val_epoch = 0
        for i_epoch in range(self.opt.num_epoch):
            logger.info(">" * 100)
            logger.info("epoch: {}".format(i_epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch["polarity"].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                batch_sz = len(outputs)
                batch_correct = (torch.argmax(outputs, -1) == targets).sum().item()
                batch_acc = 1.0*batch_correct/batch_sz
                n_correct += batch_correct
                n_total += batch_sz
                batch_loss = loss.item()
                loss_total += batch_loss * batch_sz
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info("steps: %d, total avg loss: %.4f, batch_loss: %.4f, total avg acc: %.4f, \
batch_acc: %.4f" % (global_step, train_loss, batch_loss, train_acc, batch_acc))

            val_acc, val_f1, _, _ = self._evaluate_acc_f1(val_data_loader)
            logger.info("> val_acc: {:.4f}, val_f1: {:.4f}".format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists("state_dict"):
                    os.mkdir("state_dict")
                path = "state_dict/{0}_{1}_val_acc_{2}".format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info(">> saved: {}".format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
            if i_epoch - max_val_epoch >= self.opt.patience:
                print(">> early stop.")
                break

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch["polarity"].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        gd_truths = t_targets_all.cpu()
        preds = torch.argmax(t_outputs_all, -1).cpu()
        f1 = metrics.f1_score(gd_truths, preds, labels=[0, 1, 2], average="macro")
        return acc, f1, gd_truths, preds

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        print("load model from %s" % best_model_path)
        train_acc, train_f1, _, _ = self._evaluate_acc_f1(train_data_loader)
        val_acc, val_f1, _, _ = self._evaluate_acc_f1(val_data_loader)
        test_acc, test_f1, _, _ = self._evaluate_acc_f1(test_data_loader)
        logger.info(">> train_acc: {:.4f}, train_f1: {:.4f}".format(train_acc, train_f1))
        logger.info(">> val_acc: {:.4f}, val_f1: {:.4f}".format(val_acc, val_f1))
        logger.info(">> test_acc: {:.4f}, test_f1: {:.4f}".format(test_acc, test_f1))

    def result_analysis(self, data_loader, gd_truths, preds):
        tokenizer = Tokenizer4Bert(self.opt.max_seq_len, self.opt.pretrained_bert_name)
        sentences = []
        for batch_data in data_loader:
            text_indices = batch_data["text_indices"]
            aspect_indices = batch_data["aspect_indices"]
            for text_ids, aspect_ids in zip(text_indices, aspect_indices):
                text_ids = [text_id for text_id in text_ids if text_id != 0]
                aspect_ids = [aspect_id for aspect_id in aspect_ids if aspect_id != 0]
                text = tokenizer.tokenizer.convert_ids_to_tokens(text_ids)
                aspect = tokenizer.tokenizer.convert_ids_to_tokens(aspect_ids)
                sentence = "".join(text)
                aspect = "".join(aspect)
                sentences.append((sentence, aspect))
        conf_sentence_set = [[[] for _ in range(3)] for _ in range(3)]
        for gd_idx, pred_idx, sentence in zip(gd_truths, preds, sentences):
            gd_idx = gd_idx.item()
            pred_idx = pred_idx.item()
            if gd_idx == pred_idx:
                continue
            conf_sentence_set[gd_idx][pred_idx].append(sentence)

        sentiment_lst = ["否定", "无观点", "肯定"]
        with open("test_result_analysis.txt", "w") as writer:
            cm = confusion_matrix(gd_truths, preds, labels=[0, 1, 2])
            writer.write("test confusion matrix:\n")
            writer.write("\t否定\t无观点\t肯定\n")
            writer.write("否定\t%-4d\t%-4d\t%-4d\n" % (tuple(cm[0])))
            writer.write("无观点\t%-4d\t%-4d\t%-4d\n" % (tuple(cm[1])))
            writer.write("肯定\t%-4d\t%-4d\t%-4d\n\n" % (tuple(cm[2])))
            for gd_idx in range(3):
                for pred_idx in range(3):
                    gd_senti = sentiment_lst[gd_idx]
                    pred_senti = sentiment_lst[pred_idx]
                    if len(conf_sentence_set[gd_idx][pred_idx]) == 0:
                        continue
                    writer.write("*" * 10 + " %s ==> %s " % (gd_senti, pred_senti) + "*" * 10 + "\n")
                    for sentence in conf_sentence_set[gd_idx][pred_idx]:
                        writer.write("%s\n" % str(sentence))
                    writer.write("\n\n")


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert_spc", type=str)
    parser.add_argument("--dataset", default="laptop", type=str, help="twitter, restaurant, laptop")
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--initializer", default="xavier_uniform_", type=str)
    parser.add_argument("--lr", default=2e-5, type=float, help="try 5e-5, 2e-5 for BERT, 1e-3 for others")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--l2reg", default=0.01, type=float)
    parser.add_argument("--num_epoch", default=5, type=int, help="try larger number (>20) for non-BERT models")
    parser.add_argument("--batch_size", default=32, type=int, help="try 16, 32, 64 for BERT models")
    parser.add_argument("--log_step", default=10, type=int)
    parser.add_argument("--embed_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=300, type=int)
    parser.add_argument("--bert_dim", default=768, type=int)
    parser.add_argument("--pretrained_bert_name", default="bert-base-uncased", type=str)
    parser.add_argument("--max_seq_len", default=40, type=int)
    parser.add_argument("--polarities_dim", default=3, type=int)
    parser.add_argument("--hops", default=3, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--device", default=None, type=str, help="e.g. cuda:0")
    parser.add_argument("--data_dir", default="datasets/xp/", type=str, help="xp dataset")
    parser.add_argument("--seed", default=1234, type=int, help="set seed for reproducibility")
    parser.add_argument("--valid_dataset_ratio", default=0.1, type=float,
                        help="set ratio between 0 and 1 for validation support")

    # The following parameters are only valid for the lcf-bert model
    parser.add_argument("--local_context_focus", default="cdm", type=str, help="local context focus mode, cdw or cdm")
    parser.add_argument("--SRD", default=3, type=int,
                        help="semantic-relative-distance, see the paper of LCF-BERT model")
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(opt.seed)

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
    opt.optimizer = optimizers[opt.optimizer]
    opt.inputs_cols = input_cols[opt.model_name]
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.initializer = initializers[str(opt.initializer)]
    if opt.device is None:
        if torch.cuda.is_available():
            opt.device = torch.device("cuda")
        else:
            opt.device = torch.device("cpu")
    else:
        opt.device = torch.device(str(opt.device))

    log_file = "{}-{}-{}.log".format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == "__main__":
    main()
