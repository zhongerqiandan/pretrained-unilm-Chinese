# -*- coding: utf-8 -*-
# @Time    : 2020/10/12 4:20 下午
# @Author  : Jiangweiwei
# @mail    : zhongerqiandan@163.com

from __future__ import print_function
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import re
from random import shuffle, sample
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 基本参数
maxlen = 512
batch_size = 16
epochs = 20

# bert配置
config_path = '/data/jiangweiwei/bertmodel/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/jiangweiwei/bertmodel/bert-seq2seq_mlm-lm-bi/bert_model.ckpt'
dict_path = '/data/jiangweiwei/bertmodel/chinese_L-12_H-768_A-12/vocab.txt'

rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")


def clean_text(text):
    text = rule.sub('', text)
    return text


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        json_str = f.read()
        data = json.loads(json_str, encoding='utf-8')
        for i, key in tqdm(enumerate(data)):
            question = clean_text(data[key]['question'])
            if i == 0:
                print('question:')
                print(question)
            evidences = data[key]['evidences']
            for k in evidences:
                evi = evidences[k]
                passage = clean_text(evi['evidence'])
                answer = '#'.join(evi['answer'])
                if i == 0:
                    print(passage)
                    print(answer)
                D.append((question, passage, answer))

    shuffle(D)
    return D


# 加载数据集
train_data = load_data('dataset/WebQA.v1.0/me_train.json')
valid_data = load_data('dataset/WebQA.v1.0/me_validation.ann.json')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def web_qa_tokenize(question, passage, answer, max_len):
    question_tokens = tokenizer.tokenize(question)[1:-1]
    passage_tokens = tokenizer.tokenize(passage)[1:-1]
    answer_tokens = tokenizer.tokenize(answer)[1:-1] if answer != 'no_answer' else ['[UNK]']
    first_tokens = ['[CLS]'] + question_tokens + ['[UNK]'] + passage_tokens + ['SEP']
    first_seg = [0] * len(first_tokens)
    second_tokens = answer_tokens + ['[SEP]']
    second_seg = [1] * len(second_tokens)
    if len(first_tokens + second_tokens) <= max_len:
        tokens = first_tokens + second_tokens
        segs = first_seg + second_seg
        if answer:
            return tokens, segs
        else:
            return tokens[:-1], segs[:-1]
    else:
        if answer:
            remain_len = max_len - 4
        else:
            remain_len = max_len - 3
        passage_remain_len = remain_len - len(question_tokens) - len(answer_tokens)
        if passage_remain_len > 0:
            passage_tokens = passage_tokens[:passage_remain_len]
            first_tokens = ['[CLS]'] + question_tokens + ['[UNK]'] + passage_tokens + ['SEP']
            first_seg = [0] * len(first_tokens)
            second_tokens = answer_tokens + ['[SEP]']
            second_seg = [1] * len(second_tokens)
            tokens = first_tokens + second_tokens
            segs = first_seg + second_seg
            if answer:
                return tokens, segs
            else:
                return tokens[:-1], segs[:-1]
        else:
            return None


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (question, passage, answer) in self.sample(random):
            tokens, segment_ids = web_qa_tokenize(
                question, passage, answer, max_len=maxlen
            )
            if not tokens:
                continue
            batch_token_ids.append(tokenizer.tokens_to_ids(tokens))
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path=checkpoint_path,
    application='unilm',
    model='bert',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, question, passage, topk=1):
        max_c_len = maxlen - self.maxlen
        # token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        tokens, segment_ids = web_qa_tokenize(question, passage, '', max_c_len)
        token_ids = tokenizer.tokens_to_ids(tokens)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        if not tokenizer.decode(output_ids):
            return 'no_answer'
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=16)


def show():
    samples = sample(valid_data, 6)
    for each in samples:
        question, passage, answer = each[0], each[1], each[2]
        g_answer = autotitle.generate(question, passage)
        print('-' * 36)
        print('Question:')
        print(question)
        print('Passage:')
        print(passage)
        print('Ground truth answer:')
        print(answer)
        print('Generative answer:')
        print(g_answer)


class Evaluator(keras.callbacks.Callback):
    """模型评测与保存
    """

    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            # model.save_weights('/home/jiangweiwei/pretrained-unilm-Chinese/output/webqa/best_model.weights')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
        show()

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for question, passage, answer in tqdm(data):
            total += 1
            answer = ' '.join(answer).lower()
            pred_answer = ' '.join(autotitle.generate(question, passage, topk)).lower()
            if pred_answer.strip():
                scores = self.rouge.get_scores(hyps=pred_answer, refs=answer)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[answer.split(' ')],
                    hypothesis=pred_answer.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
