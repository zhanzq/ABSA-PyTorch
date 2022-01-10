# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com
# date  : 2021/11/3
#

from evaluate import Inference
from config import option, id2senti


def main():
    inf = Inference(opt=option)
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
        t_probs = inf.infer(text=test_sentence, aspect=aspect)
        label = t_probs.argmax(axis=-1)
        confidence = t_probs[label]
        sentiment = id2senti[label]
        print("(%s, %s, %s), confidence=%.4f" % (test_sentence, aspect, sentiment, confidence))
        prompt = "请输入用户语句:"
        print(prompt)
        test_sentence = input()


if __name__ == '__main__':
    main()
