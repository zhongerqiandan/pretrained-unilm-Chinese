# -*- coding: utf-8 -*-
# @Time    : 2020/10/24 1:50 下午
# @Author  : Jiangweiwei
# @mail    : zhongerqiandan@163.com

import sys
import copy
import codecs
import pickle
import random
import functools
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer

from pathlib import Path
from random import shuffle, sample

PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

dict_path = '/data/jiangweiwei/bertmodel/chinese_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
with open(dict_path, 'r') as f:
    tokens = f.readlines()


def parse_data(path):
    """process the data."""
    with open(path, 'r') as file:
        lines = file.readlines()
        shuffle(lines)
        questions = []
        answers = []
        segment_ids = []
        for line in tqdm(lines):
            line = line.strip().split('\t')
            que, ans = line[0], line[1]
            que = tokenizer.tokenize(que)
            ans = tokenizer.tokenize(ans)
            que = tokenizer.tokens_to_ids(que)
            ans = tokenizer.tokens_to_ids(ans)
            ans = ans[1:]
            segment_id = [0] * len(que) + [1] * len(ans)
            questions.append(que)
            answers.append(ans)
            segment_ids.append(segment_id)
    assert len(questions) == len(answers)

    return questions, answers, segment_ids


def create_attention_mask_for_seq(segment_id, input_mask):
    attention_mask = np.zeros(shape=(len(segment_id), len(segment_id)))
    attention_mask += np.array(input_mask)

    first_len = len(segment_id) - sum(segment_id)
    for i in range(len(segment_id) - 1):
        if i < first_len:
            attention_mask[i][first_len:] = 0
        else:
            attention_mask[i][i + 1:] = 0
    return attention_mask


def create_attention_mask_for_lm(input_mask, reverse=False):
    attention_mask = np.zeros(shape=(len(input_mask), len(input_mask)))
    attention_mask += np.array(input_mask)
    for i in range(len(input_mask) - 1):
        attention_mask[i][i + 1:] = 0
    if not reverse:
        return attention_mask
    input_mask.reverse()
    return np.flipud(np.fliplr(create_attention_mask_for_lm(input_mask)))


def create_attention_mask_for_bi(input_mask):
    attention_mask = np.zeros(shape=(len(input_mask), len(input_mask)))
    attention_mask += np.array(input_mask)
    return attention_mask


def creat_input_mask(input_id, max_len, p=0.15):
    '''

    :param input_id: [cls_id, id1, id2, id3,..., sep_id]
    :param max_len:
    :param p:
    :return:
    '''
    if len(input_id) > max_len - 2:
        true_ids = input_id[1:max_len - 1]
        input_mask = [1] * len(true_ids)
        masked_ids = []
        masked_positions = []
        randns = np.random.random((len(true_ids)))
        for i in range(len(true_ids)):
            if randns[i] <= p:
                masked_ids.append(true_ids[i])
                masked_positions.append(i + 1)
                randn = np.random.random(1)[0]
                if randn <= 0.8:
                    true_ids[i] = tokenizer.tokens_to_ids(['[MASK]'])[0]
                elif randn > 0.8 and randn <= 0.9:
                    token = sample(tokens, 1)[0]
                    true_ids[i] = tokenizer.tokens_to_ids([token])[0]
                input_mask[i] = 0
                # true_ids[i] = tokenizer.tokens_to_ids(['[MASK]'])[0]
        # [CLS]和[SEP]
        input_mask = [1] + input_mask + [1]
        input_id = [tokenizer.tokens_to_ids(['[CLS]'])[0]] + true_ids + [tokenizer.tokens_to_ids(['[SEP]'])[0]]
        masked_weights = [1] * len(masked_positions) + [0] * (max_len - len(masked_positions))
        masked_positions = masked_positions + [0] * (max_len - len(masked_positions))
        # 因为masked_positions的补齐位置是0, 0位置的id是[CLS]的id
        masked_ids = masked_ids + [tokenizer.tokens_to_ids(['[CLS]'])[0]] * (max_len - len(masked_ids))
        return input_id, input_mask, masked_ids, masked_positions, masked_weights
    else:
        true_ids = input_id[1:-1]
        true_ids_paded = true_ids + [tokenizer.tokens_to_ids(['[PAD]'])[0]] * (max_len - 2 - len(true_ids))
        input_mask = [1] * len(true_ids) + [0] * (max_len - 2 - len(true_ids))
        masked_ids = []
        masked_positions = []
        randns = np.random.random((len(true_ids)))
        for i in range(len(true_ids)):
            if randns[i] <= p:
                masked_ids.append(true_ids[i])
                masked_positions.append(i + 1)
                randn = np.random.random(1)[0]
                if randn <= 0.8:
                    true_ids_paded[i] = tokenizer.tokens_to_ids(['[MASK]'])[0]
                elif randn > 0.8 and randn <= 0.9:
                    token = sample(tokens, 1)[0]
                    true_ids_paded[i] = tokenizer.tokens_to_ids([token])[0]
                input_mask[i] = 0
        input_mask = [1] + input_mask + [1]
        input_id = [tokenizer.tokens_to_ids(['[CLS]'])[0]] + true_ids_paded + [tokenizer.tokens_to_ids(['[SEP]'])[0]]
        masked_weights = [1] * len(masked_positions) + [0] * (max_len - len(masked_positions))
        masked_positions = masked_positions + [0] * (max_len - len(masked_positions))
        # 因为masked_positions的补齐位置是0, 0位置的id是[CLS]的id
        masked_ids = masked_ids + [tokenizer.tokens_to_ids(['[CLS]'])[0]] * (max_len - len(masked_ids))
        return input_id, input_mask, masked_ids, masked_positions, masked_weights


def train_generator(path, max_length):
    """"This is the entrance to the input_fn."""
    questions, answers, segment_ids = parse_data(path)
    randns = np.random.random((len(questions)))

    for que, ans, segment_id, randn in zip(questions, answers, segment_ids, randns):
        if randn < 0.34:
            input_id = que + ans
            if len(segment_id) - sum(segment_id) >= max_length:
                # 第一个句子长度大于max_length
                continue
            input_id, input_mask, masked_ids, masked_positions, masked_weights = creat_input_mask(input_id, max_length)
            segment_id += [1] * (max_length - len(segment_id))
            segment_id = segment_id[:max_length]
            attention_mask = create_attention_mask_for_seq(segment_id, input_mask)
        elif randn >= 0.34 and randn <= 0.67:
            input_id = que + ans
            input_id, input_mask, masked_ids, masked_positions, masked_weights = creat_input_mask(input_id, max_length)
            attention_mask = create_attention_mask_for_bi(input_mask)
            segment_id += [1] * (max_length - len(segment_id))
            segment_id = segment_id[:max_length]
        elif randn > 0.67 and randn <= 0.83:
            input_id = que + ans
            input_id, input_mask, masked_ids, masked_positions, masked_weights = creat_input_mask(input_id, max_length)
            segment_id += [1] * (max_length - len(segment_id))
            segment_id = segment_id[:max_length]
            attention_mask = create_attention_mask_for_lm(input_mask)
        else:
            input_id = que + ans
            input_id, input_mask, masked_ids, masked_positions, masked_weights = creat_input_mask(input_id, max_length)
            segment_id += [1] * (max_length - len(segment_id))
            segment_id = segment_id[:max_length]
            attention_mask = create_attention_mask_for_lm(input_mask, reverse=True)

        features = {'input_ids': input_id,
                    'input_mask': attention_mask,
                    'segment_ids': segment_id,
                    'masked_lm_positions': masked_positions,
                    'masked_lm_ids': masked_ids,
                    'masked_lm_weights': masked_weights}
        assert len(features['input_ids']) == len(features['input_mask']) == len(features['segment_ids']) == len(
            features['masked_lm_positions']) == len(features['masked_lm_ids']) == len(
            features['masked_lm_weights']) == max_length
        yield features


def train_input_fn(params):
    path = params['path']
    batch_size = params['batch_size']
    repeat_num = params['repeat_num']
    max_length = params['max_length']

    output_types = {'input_ids': tf.int32,
                    'input_mask': tf.int32,
                    'segment_ids': tf.int32,
                    'masked_lm_positions': tf.int32,
                    'masked_lm_ids': tf.int32,
                    'masked_lm_weights': tf.float32}
    output_shape = {'input_ids': [None],
                    'input_mask': [None, None],
                    'segment_ids': [None],
                    'masked_lm_positions': [None],
                    'masked_lm_ids': [None],
                    'masked_lm_weights': [None]}

    dataset = tf.data.Dataset.from_generator(
        functools.partial(train_generator, path, max_length),
        output_types=output_types,
        output_shapes=output_shape)
    dataset = dataset.batch(batch_size).repeat(repeat_num)

    return dataset


def serving_input_receiver_fn():
    """For prediction input."""
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='input_mask')
    masked_lm_positions = tf.placeholder(dtype=tf.int32, shape=[None, None], name='masked_lm_postions')

    receiver_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'masked_lm_positions': masked_lm_positions}
    features = {'input_ids': input_ids,
                'input_mask': input_mask,
                'masked_lm_positions': masked_lm_positions,
                'masked_lm_ids': tf.zeros([1, 10], dtype=tf.int32),
                'masked_lm_weights': tf.zeros([1, 10], dtype=tf.int32)}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
