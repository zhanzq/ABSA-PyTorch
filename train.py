# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/3
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
        self.n_total, self.global_step, self.n_correct, self.max_valid_epoch = 0, 0, 0, 0
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

    def train(self,):
        train_data_loader = DataLoader(dataset=self.train_dataset, num_workers=4,
                                       batch_size=self.opt.batch_size, shuffle=True)
        valid_data_loader = DataLoader(dataset=self.valid_dataset, num_workers=4, batch_size=self.opt.eval_batch_size)

        self._reset_params()
        self.global_step, self.n_correct, self.n_total, self.loss_total = 0, 0, 0, 0

        start_time = time.time()
        for i_epoch in range(self.opt.num_epoch):
            logger.info("*" * 40 + "epoch: {:-2d}".format(i_epoch) + "*" * 40)
            # switch model to training mode
            self.model.train()

            for i_batch, batch in enumerate(train_data_loader):
                self.global_step += 1
                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = batch["polarity"].to(self.opt.device)
                batch_correct, batch_loss = self.train_step(inputs, targets)
                batch_size = len(targets)

                if self.global_step % self.opt.log_step == 0:
                    end_time = time.time()
                    self.do_log_step(batch_correct, batch_loss, batch_size, start_time, end_time)
                    start_time = end_time

            valid_res = self.evaluate_acc_f1(valid_data_loader)
            self.valid_acc, self.valid_f1 = valid_res["acc"], valid_res["f1"]
            logger.info("> valid_acc: {:.4f}, valid_f1: {:.4f}".format(self.valid_acc, self.valid_f1))

            self.update_model(cur_epoch=i_epoch)
            if self.early_stop:
                break

    def do_log_step(self, batch_correct, batch_loss, batch_size, start_time, end_time):
        self.n_total += batch_size
        self.loss_total += batch_loss * batch_size
        self.n_correct += batch_correct
        avg_loss = self.loss_total / self.n_total
        avg_acc = 1.0 * self.n_correct / self.n_total
        batch_acc = 1.0 * batch_correct/batch_size

        logger.info("steps: %d, ttl_avg_loss: %.4f, batch_loss: %.4f, ttl_avg_acc: %.4f, batch_acc: %.4f, cost: %.3f s"
                    % (self.global_step, avg_loss, batch_loss, avg_acc, batch_acc, end_time - start_time))

    def train_step(self, inputs, targets):

        # clear gradient accumulators
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        batch_correct = (torch.argmax(outputs, -1) == targets).sum().item()
        batch_loss = loss.item()

        return batch_correct, batch_loss

    def update_model(self, cur_epoch):
        if self.valid_acc <= self.max_valid_acc:
            return

        # save new better model
        if not os.path.exists("state_dict"):
            os.mkdir("state_dict")
        save_path = "state_dict/{0}_{1}_{2}_{3}.pt".format(self.opt.arch_name, self.model_name,
                                                           "%.4f" % self.valid_acc, self.run_tag)
        torch.save(self.model.state_dict(), save_path)
        logger.info(">> saved better model: {}".format(save_path))
        self.opt.best_model_path = save_path

        org_path = "state_dict/{0}_{1}_{2}_{3}.pt".format(self.opt.arch_name, self.model_name,
                                                          "%.4f" % self.max_valid_acc, self.run_tag)
        if os.path.exists(org_path):
            os.remove(org_path)
            logger.info(">> remove older model: {}".format(org_path))

        if cur_epoch - self.max_valid_epoch >= self.opt.patience:
            logger.info(">> early stop.")
            self.early_stop = True

        self.max_valid_acc = self.valid_acc
        self.max_valid_epoch = cur_epoch

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
            acc, f1 = res["acc"], res["f1"]
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
        logger.info(">> test_acc: {:.4f}, test_f1: {:.4f}".format(test_res["acc"], test_res["f1"]))

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
        f1 = metrics.f1_score(gd_truths, preds, labels=[0, 1, 2], average="macro")

        outputs = {
            "acc": acc,
            "f1": f1,
            "gd_truths": gd_truths,
            "preds": preds,
        }

        return outputs

    def run(self, run_tag):
        self.run_tag = run_tag
        self.train()
        eval_res = self.evaluate()
        result = [eval_res["train_acc"], eval_res["train_f1"], eval_res["valid_acc"],
                  eval_res["valid_f1"], eval_res["test_acc"], eval_res["test_f1"]]

        return result

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


def record_train_results(opt, results):
    with open(opt.train_res_path, "a") as writer:
        model_name = opt.pretrained_bert_name.split("/")[-1]
        arch_name = opt.arch_name
        avg_result = numpy.mean(results, axis=0)
        avg_result = [it.item() for it in avg_result]
        # write logs
        writer.write("pretrained model: {:s}, model architecture: {:s}\n".format(model_name, arch_name))
        writer.write("{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}\n".format("run_idx", "train_acc", "train_f1",
                                                                           "valid_acc", "valid_f1",
                                                                           "test_acc", "test_f1"))
        for i, result_i in enumerate(results):
            writer.write("%-10d%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f\n" % tuple([i] + result_i))
        writer.write("%-10s%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f%-10.4f\n\n\n" % tuple(["avg"] + avg_result))


def main():
    log_file = "{}-{}-{}.log".format(option.arch_name, option.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    if option.do_train:
        run_num = 5
        results = []
        for i in range(run_num):
            ins = Instructor(opt=option)
            result_i = ins.run(run_tag=str(i))
            results.append(result_i)
        record_train_results(opt=option, results=results)

    if option.do_eval:
        ins = Instructor(opt=option)
        ins.evaluate()

    if option.do_test:
        ins = Instructor(opt=option)
        ins.test()


if __name__ == "__main__":
    main()
