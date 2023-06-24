import os
import sys
if "peft" not in os.getcwd():
    os.chdir("peft")
sys.path.append(os.getcwd())
import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.data.datasets import *
from datasets import load_dataset
from src.superglue_loader import get_superglue


class SuperGlueData(Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_length=512, text_to_text=False):
        super().__init__()
        # ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
        self.dataset = []
        self.split = split
        self.data_list, self.num_labels = get_superglue(dataset_name, split, text_to_text)     
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_to_text = text_to_text


    def create_dataset(self, debug=False):
        zeros, ones = 0, 0
        if debug:
            self.data_list = self.data_list[:64]
        for i, data in tqdm(enumerate(self.data_list), total=len(self.data_list), desc="Formatting dataset", ncols=100):
            if len(data) == 2:
                input_seq, label = data
            elif len(data) == 3:
                input_seq, label, _ = data
            input_ids, segment_ids, label= self.formatting(input_seq, label)
            if input_ids is not None:
                self.dataset.append({"ids": i,
                                    "input_ids": input_ids,
                                    "segment_ids": segment_ids,
                                    "label": label})
        random.shuffle(self.dataset)

    def formatting(self, input_seq, label):
        if type(input_seq) == tuple:
            s0, s1 = input_seq
            if type(s0) == tuple:
                s00, s01 = s0
                s10, s11 = s1
                input_ids_0, segment_ids_0 = self.get_input_ids_from_tuple(s00, s01)
                input_ids_1, segment_ids_1 = self.get_input_ids_from_tuple(s10, s11)
                len0, len1 = len(input_ids_0), len(input_ids_1)
                if len0 > len1:
                    diff = len0 - len1
                    input_ids_1 += [self.tokenizer.pad_token_id] * diff
                    segment_ids_1 += [0] * diff
                elif len1 > len0:
                    diff = len1 - len0
                    input_ids_0 += [self.tokenizer.pad_token_id] * diff         
                    segment_ids_0 += [0] * diff
                input_ids = [input_ids_0, input_ids_1]
                segment_ids = [segment_ids_0, segment_ids_1]
            else:
                input_ids, segment_ids = self.get_input_ids_from_tuple(s0, s1)
        else:
            token_ids_0 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_seq))            
            # input_ids = token_ids_0
            input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0)           
            segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0)  

        # assert len(segment_ids) == len(input_ids)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        if self.text_to_text:
            label = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(f"{label}{self.tokenizer.eos_token}"))
            label = torch.LongTensor(label)

        input_ids = torch.LongTensor(input_ids)
        segment_ids = torch.LongTensor(segment_ids)
        return input_ids, segment_ids, label

    def get_input_ids_from_tuple(self, s0, s1):
        token_ids_0 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s0))
        token_ids_1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(s1))
        input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        segment_ids = self.tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)    

        return input_ids, segment_ids
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_num_labels(self):
        return self.num_labels
