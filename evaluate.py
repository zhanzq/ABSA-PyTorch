# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/12/27
#

import os
import sys
import math
import time

import numpy
import logging

from sklearn import metrics
from time import strftime, localtime
from sklearn.metrics import confusion_matrix

from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import option
from data_utils import load_data
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:

    def __init__(self, opt):
        self.opt = opt
        self.run_tag = ""
        self.early_stop = False
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = opt.pretrained_bert_name.split("/")[-1]
        self.max_valid_acc, self.valid_acc, self.valid_f1, self.loss_total = 0.0, 0.0, 0.0, 0.0

        if "bert" in opt.arch_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            if "electra" in opt.pretrained_bert_name:
                dct = torch.load(os.path.join(opt.pretrained_bert_name, "pytorch_model.bin"))
                state_dict = {}
                for key in dct:
                    if key.startswith("electra."):
                        state_dict[key[8:]] = dct[key]
                    elif key.startswith("discriminator_predictions."):
                        prefix_len = len("discriminator_predictions.")
                        _key = "pooler." + key[prefix_len:]
                        state_dict[_key] = dct[key]
                bert = BertModel.from_pretrained(opt.pretrained_bert_name, state_dict=state_dict)
            else:
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

        # Loss and Optimizer
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data, valid_data, test_data = load_data(data_dir=opt.data_dir,
                                                      norm_text=False, tokenizer=tokenizer)

        self.train_dataset = ABSADataset(train_data, tokenizer)
        self.valid_dataset = ABSADataset(valid_data, tokenizer)
        self.test_dataset = ABSADataset(test_data, tokenizer)

        if opt.device.type == "cuda":
            logger.info("cuda memory allocated: {}".format(torch.cuda.memory_allocated(device=opt.device.index)))
        self.print_args()

    def print_args(self):
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

    def evaluate(self):
        # load model
        assert os.path.exists(self.opt.best_model_path), "pretrained model must exist for evaluation"
        self.model.load_state_dict(torch.load(self.opt.best_model_path, map_location=self.opt.device))
        logger.info("load model from %s" % self.opt.best_model_path)

        # construct data loader
        train_data_loader = DataLoader(dataset=self.train_dataset, batch_size=self.opt.eval_batch_size, num_workers=4)
        test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.eval_batch_size, num_workers=4)
        valid_data_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.opt.eval_batch_size, num_workers=4)

        # calculate accuracy and f1
        train_res = self.evaluate_acc_f1(train_data_loader)
        valid_res = self.evaluate_acc_f1(valid_data_loader)
        test_res = self.evaluate_acc_f1(test_data_loader)
        results = [train_res, valid_res, test_res]
        tags = ["train", "valid", "test"]
        eval_res = {}
        for tag, res in zip(tags, results):
            acc, f1 = res["acc"], numpy.mean(res["f1_lst"])
            eval_res["%s_acc" % tag], eval_res["%s_f1" % tag] = acc, f1
            logger.info(">> {:s}: acc={:.4f}, f1={:.4f}".format(tag, acc, f1))

        return eval_res

    def test(self):
        # load model
        assert os.path.exists(self.opt.best_model_path), "pretrained model must exist for test"
        self.model.load_state_dict(torch.load(self.opt.best_model_path))
        logger.info("load model from %s" % self.opt.best_model_path)

        # construct data loader
        test_data_loader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.eval_batch_size, num_workers=4)

        # calculate accuracy and f1
        test_res = self.evaluate_acc_f1(test_data_loader)
        acc, f1 = test_res["acc"], numpy.mean(test_res["f1"])
        logger.info(">> test_acc: {:.4f}, test_f1: {:.4f}".format(acc, f1))

        # result analysis
        self.result_analysis(test_data_loader, gd_truths=test_res["gd_truths"], preds=test_res["preds"])

        return test_res

    def evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        gd_truths, preds = [], []

        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                batch_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                batch_targets = t_batch["polarity"].to(self.opt.device)
                batch_outputs = self.model(batch_inputs)
                batch_preds = torch.argmax(batch_outputs, -1)

                n_correct += (batch_targets == batch_preds).sum().item()
                n_total += len(batch_preds)

                gd_truths.extend([it.item() for it in batch_targets])
                preds.extend([it.item() for it in batch_preds])

        acc = n_correct / (n_total + 1.0e-5)
        acc_lst = metrics.accuracy_score(gd_truths, preds)
        p_lst = metrics.precision_score(gd_truths, preds, labels=[0, 1, 2], average=None)
        r_lst = metrics.recall_score(gd_truths, preds, labels=[0, 1, 2], average=None)
        f1_lst = metrics.f1_score(gd_truths, preds, labels=[0, 1, 2], average=None)

        outputs = {
            "acc": acc,
            "p_lst": p_lst,
            "r_lst": r_lst,
            "f1_lst": f1_lst,
            "gd_truths": gd_truths,
            "preds": preds,
        }

        return outputs

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
    log_file = "{}-{}-{}.log".format(option.arch_name, option.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    if option.do_eval:
        logger.info(msg="evaluate train, valid, test datasets ...")
        ins = Instructor(opt=option)
        ins.evaluate()

    if option.do_test:
        logging.log(msg="evaluate test dataset and analyze the results ...")
        ins = Instructor(opt=option)
        ins.test()


if __name__ == "__main__":
    main()
