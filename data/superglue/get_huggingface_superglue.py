import os
import sys
if "SMoP" not in os.getcwd():
    os.chdir("SMoP")
from datasets import load_dataset, list_datasets
import pickle

# CoLA SST MRPC STS QQP MNLI QNLI RTE

dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic']

for dataset in dataset_names:
    os.makedirs(f"data/superglue/{dataset}", exist_ok=True)
    data = load_dataset('super_glue', dataset)
    for key in data.keys():
        path = f"data/superglue/{dataset}/{key}.pkl"
        # d = load_dataset('glue', dataset)[key]
        # for x in d:
        #     print(x)
        pickle.dump(data[key], open(path, 'wb'))


