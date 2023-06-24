import os
import sys
if "peft" not in os.getcwd():
    os.chdir("peft")
sys.path.append(os.getcwd())

import random
from tqdm import tqdm
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.data.datasets import *
from datasets import load_dataset
from collections import defaultdict


def construct_dev_from_train_t2t(train_list, num_labels):
    if num_labels == 1:
        border = len(train_list) * 8 // 9
        val_list = train_list[border:]
        actual_train_list = train_list[:border]
    else:
        actual_train_list = []
        val_list = []
        label_dict = defaultdict(list)
        for input_text, label_text in train_list:
            label_dict[label_text].append(input_text)
        
        for label_text in label_dict:
            inputs = label_dict[label_text]
            n_inputs = len(inputs)
            border = n_inputs * 9 // 10
            actual_train_list.extend([(i, label_text) for i in inputs[:border]])
            val_list.extend([(i, label_text) for i in inputs[border:]])

    return actual_train_list, val_list


def boolq(text_to_text):
    path = "data/superglue/boolq"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        # https://huggingface.co/datasets/stjokerli/TextToText_boolq/viewer/stjokerli--TextToText_boolq/train
        text_format = f"boolq passage: **passage** question: **question**"
        labels = ("False", "True")
        train_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**passage**", d['passage']).replace("**question**", d['question']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        train_list = [((d['question'], d['passage']), d['label']) for d in train]
        val_list = [((d['question'], d['passage']), d['label']) for d in val]
        num_labels = max(train['label']) + 1

    return train_list, val_list, num_labels


def cb(text_to_text):
    path = "data/superglue/cb"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = f"cb hyopthesis: **hypothesis**. premise: **premise**"
        labels = ('entailment', 'contradiction', 'neutral')
        train_list = [(text_format.replace("**hypothesis**", d['hypothesis']).replace("**premise**", d['premise']), labels[d['label']]) for d in train]
        val_list =  [(text_format.replace("**hypothesis**", d['hypothesis']).replace("**premise**", d['premise']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        train_list = [((d['premise'], d['hypothesis']), d['label']) for d in train]
        val_list =  [((d['premise'], d['hypothesis']), d['label']) for d in val]
        num_labels = max(train['label']) + 1
    return train_list, val_list, num_labels

def copa(text_to_text):
    path = "data/superglue/copa"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = "copa choice1: **choice1** choice2: **choice2** premise: **premise** question: **question**"
        labels = ('choice1', 'choice2')
        train_list = [(f"choice1: {d['choice1']} choice2: {d['choice2']} premise: {d['premise']} question: {d['question']}", labels[d['label']]) for d in train]
        val_list = [(f"choice1: {d['choice1']} choice2: {d['choice2']} premise: {d['premise']} question: {d['question']}", labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        train_list, val_list = []
        for d in train:
            joiner = 'beacuse' if d['question'] == 'cause' else 'so'
            text_a = f"{d['premise']} {joiner}"
            train_list.append((((text_a, d['choice1']), (text_a, d['choice2'])), d['label']))
        for d in val:
            joiner = 'beacuse' if d['question'] == 'cause' else 'so'
            text_a = f"{d['premise']} {joiner}"
            val_list.append((((text_a, d['choice1']), (text_a, d['choice2'])), d['label']))
        num_labels = max(train['label']) + 1
    return train_list, val_list, num_labels


def multirc(text_to_text):
    path = "data/superglue/multirc"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = "multirc question: **question** answer: **answer**. paragraph: **paragraph**"
        labels = ["False", "True"]
        train_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", labels[d['label']]) for d in train]
        val_list = [(f"question: {d['question']} answer: {d['answer']}. paragraph: {d['paragraph']}", labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        train_list = [((d['paragraph'], f"{d['question']} {d['answer']}"), d['label']) for d in train]
        val_list = [((d['paragraph'], f"{d['question']} {d['answer']}"), d['label']) for d in val]
        num_labels = max(train['label']) + 1

    return train_list, val_list, num_labels


def rte(text_to_text):
    path = "data/superglue/rte"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        # text_format = "rte sentence1: **premise** sentence2: **hypothesis**"
        text_format = "rte sentence1: **premise** sentence2: **hypothesis**"
        labels = ("entailment", "not_entailment")
        train_list = [(text_format.replace("**premise**", d['premise']).replace("**hypothesis**", d['hypothesis']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**premise**", d['premise']).replace("**hypothesis**", d['hypothesis']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        train_list = [((d['premise'], d['hypothesis']), d['label']) for d in train]
        val_list =  [((d['premise'], d['hypothesis']), d['label']) for d in val]
        num_labels = max(train['label']) + 1
    
    random.seed(42)
    random.shuffle(train_list)
    return train_list, val_list, num_labels

def wic(text_to_text):
    path = "data/superglue/wic"
    train = pickle.load(open(os.path.join(path, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(path, 'validation.pkl'), 'rb'))

    if text_to_text:
        text_format = "wic sentence1: **sentence1** sentence2: **sentence2** word: **word**"
        labels = ('False', 'True')
        train_list = [(text_format.replace("**sentence1**", d['sentence1']).replace("**sentence2**", d['sentence2']).replace("**word**", d['word']), labels[d['label']]) for d in train]
        val_list = [(text_format.replace("**sentence1**", d['sentence1']).replace("**sentence2**", d['sentence2']).replace("**word**", d['word']), labels[d['label']]) for d in val]
        num_labels = len(labels)
    else:
        train_list = [(f"{d['sentence1']} {d['sentence2']} Does {d['word']} have the same meaning in both sentences?", d['label']) for d in train]
        val_list = [(f"{d['sentence1']} {d['sentence2']} Does {d['word']} have the same meaning in both sentences?", d['label']) for d in val]
        num_labels = max(train['label']) + 1
    return train_list, val_list, num_labels



def get_superglue(data_name, split, text_to_text=False):
    # ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']

    # Define a dictionary that maps data names to corresponding functions
    data_funcs = {'boolq': boolq, 'cb': cb, 'copa': copa, 'multirc': multirc, 'rte_superglue': rte, 'wic': wic}
    if data_name not in data_funcs:
        raise ValueError(f"Invalid data_name '{data_name}'.")

    if data_name == 'semeval':
        train_list, val_list, test_list, num_labels = data_funcs[data_name](text_to_text)
    else:
        train_list, val_list, num_labels = data_funcs[data_name](text_to_text)
        if text_to_text:
            test_list = val_list
            train_list, val_list = construct_dev_from_train_t2t(train_list, num_labels)
            
            # val_list, test_list = construct_test_from_dev_t2t(val_list, num_labels)
        else:
            val_list, test_list = construct_test_from_dev(val_list, num_labels)

    if split == "train":
        return train_list, num_labels
    elif split == "dev":
        return val_list, num_labels
    elif split == "test":
        return test_list, num_labels
    else:
        raise ValueError