# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2021/11/3
#

import os
import re
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import time
from functools import wraps


def time_cost(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start_time = time.time()
        ret_res = f(*args, **kwargs)
        end_time = time.time()
        print("%s cost time: %.3f seconds" % (f.__name__, end_time - start_time))
        return ret_res

    return decorated


def build_tokenizer(data_files, max_seq_len, tokenizer_file):
    if os.path.exists(tokenizer_file):
        print('loading tokenizer:', tokenizer_file)
        tokenizer = pickle.load(open(tokenizer_file, 'rb'))
    else:
        text = ''
        for data_file in data_files:
            fin = open(data_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_i = lines[i].lower().strip()
                aspect = lines[i+1].lower().strip()
                text_left, _, text_right = [s.lower().strip() for s in text_i.partition("$T$")]
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(tokenizer_file, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, embedding_file):
    if os.path.exists(embedding_file):
        print('loading embedding_matrix:', embedding_file)
        embedding_matrix = pickle.load(open(embedding_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        embedding_file = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(embedding_file, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_file)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_file, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, max_len, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(max_len) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-max_len:]
    else:
        trunc = sequence[:max_len]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unk_idx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unk_idx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


def rm_punctuation(text):
    punctuations = [",", "，", "\.", "。", "\?", "？", ":", "：", "!", "！", "'", "\"", "、"]
    for pun in punctuations:
        text = re.sub(pun, "", text)
    return text


def get_example(utt, aspect, polar=None, with_punctuation=False, tokenizer=None):
    polarity = None
    if polar is not None:
        polarity = int(polar) + 1
    if not with_punctuation:
        utt = rm_punctuation(text=utt)
    text_left, _, text_right = [s.strip() for s in utt.partition(aspect)]

    # if aspect != "[UNK]":
    #     utt = text_left + "[unused1]" + text_right
    #     aspect = "[unused1]"
    text_indices = tokenizer.text_to_sequence(utt)
    aspect_indices = tokenizer.text_to_sequence(aspect)
    left_indices = tokenizer.text_to_sequence(text_left)
    right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
    context_indices = tokenizer.text_to_sequence(text_left + text_right)
    left_with_aspect_indices = tokenizer.text_to_sequence(text_left + aspect)
    right_with_aspect_indices = tokenizer.text_to_sequence(aspect + text_right, reverse=True)

    left_len = np.sum(left_indices != 0)
    text_len = np.sum(text_indices != 0)
    aspect_len = np.sum(aspect_indices != 0)
    aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

    concat_bert_indices = tokenizer.text_to_sequence('[CLS]' + utt + '[SEP]' + aspect + "[SEP]")
    concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
    concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

    text_bert_indices = tokenizer.text_to_sequence("[CLS]" + utt + "[SEP]")
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS]" + aspect + "[SEP]")

    data = {
        'polarity': polarity,
        'text_indices': text_indices,
        'left_indices': left_indices,
        'right_indices': right_indices,
        'aspect_indices': aspect_indices,
        'context_indices': context_indices,
        'aspect_boundary': aspect_boundary,
        'text_bert_indices': text_bert_indices,
        'aspect_bert_indices': aspect_bert_indices,
        'concat_bert_indices': concat_bert_indices,
        'concat_segments_indices': concat_segments_indices,
        'left_with_aspect_indices': left_with_aspect_indices,
        'right_with_aspect_indices': right_with_aspect_indices,
    }

    return data


@time_cost
def get_dataset(lines, with_punctuation=False, tokenizer=None):
    all_data = []
    for line in lines:
        items = line.split("\t")
        assert len(items) == 3, "valid record must contain 3 columns, bad record: {:s}".format(line)
        utt, aspect, polar = items
        # data = get_example(utt=utt, aspect=aspect, polar=polar, with_punctuation=with_punctuation, tokenizer=tokenizer)
        data = {
            "utt": utt,
            "aspect": aspect,
            "polar": polar,
        }
        all_data.append(data)
        # yield data

    return all_data


@time_cost
def load_data(data_dir, with_punctuation=False, tokenizer=None, do_shuffle=True, splits=(0.7, 0.2, 0.1)):
    train_lines, valid_lines, test_lines = [], [], []
    assert os.path.isdir(data_dir), "input path must be data dir"
    files = os.listdir(data_dir)
    for file_name in files:
        data_path = os.path.join(data_dir, file_name)
        if "train.txt" in file_name:
            with open(data_path, "r", encoding="utf-8", newline="\n", errors="ignore") as reader:
                lines = reader.readlines()
                train_lines.extend(lines)
        elif "valid.txt" in file_name:
            with open(data_path, "r", encoding="utf-8", newline="\n", errors="ignore") as reader:
                lines = reader.readlines()
                valid_lines.extend(lines)
        elif "test.txt" in file_name:
            with open(data_path, "r", encoding="utf-8", newline="\n", errors="ignore") as reader:
                lines = reader.readlines()
                test_lines.extend(lines)
        else:
            print("irrelevant data file: {:s}".format(data_path))
    train_sz = len(train_lines)
    if do_shuffle:
        random.shuffle(train_lines)
    total_sz = train_sz
    if len(test_lines) == 0:
        test_sz = int(total_sz*splits[2])
        train_sz -= test_sz
        test_lines = train_lines[-test_sz:]
    if len(valid_lines) == 0:
        valid_sz = int(total_sz*splits[1])
        train_sz -= valid_sz
        valid_lines = train_lines[train_sz:train_sz + valid_sz]

    train_data = get_dataset(train_lines, with_punctuation=with_punctuation, tokenizer=tokenizer)
    valid_data = get_dataset(valid_lines, with_punctuation=with_punctuation, tokenizer=tokenizer)
    test_data = get_dataset(test_lines, with_punctuation=with_punctuation, tokenizer=tokenizer)

    return train_data, valid_data, test_data


class ABSADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data_i = self.data[index]
        utt = data_i["utt"]
        aspect = data_i["aspect"]
        polar = data_i["polar"]
        return get_example(utt, aspect, polar=polar, tokenizer=self.tokenizer, with_punctuation=False)
        # return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    pretrained_model_path = "/data/zhanzhiqiang/models/chinese-bert-base"
    pretrained_model_path = "/Users/zhanzq/Downloads/models/bert-base-chinese"
    tokenizer = Tokenizer4Bert(max_seq_len=40, pretrained_bert_name=pretrained_model_path)
    train_dataset, valid_dataset, test_dataset = load_data(data_dir="./datasets/xp/", with_punctuation=False,
                                                           tokenizer=tokenizer, do_shuffle=True, splits=(0.7, 0.2, 0.1))

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
