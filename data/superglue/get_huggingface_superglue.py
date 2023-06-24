import os
import sys
if "peft" not in os.getcwd():
    os.chdir("peft")
from datasets import load_dataset, list_datasets
import pickle

# CoLA SST MRPC STS QQP MNLI QNLI RTE

dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc.fixed']
# print(data)

# for dataset in dataset_names:
#     os.makedirs(f"data/superglue/{dataset}", exist_ok=True)
#     data = load_dataset('super_glue', dataset)
#     for key in data.keys():
#         path = f"data/superglue/{dataset}/{key}.pkl"
#         # d = load_dataset('glue', dataset)[key]
#         # for x in d:
#         #     print(x)
#         pickle.dump(data[key], open(path, 'wb'))



#data = load_dataset("stjokerli/TextToText_record_seqio")
# dir_path = f"data/superglue/record_text_to_text"
data = load_dataset('super_glue', 'wsc.fixed')
dir_path = f"data/superglue/wsc"
os.makedirs(dir_path, exist_ok=True)
for key in data.keys():
    path = os.path.join(dir_path, f"{key}.pkl")
    pickle.dump(data[key], open(path, 'wb'))

# data = load_dataset('super_glue', 'wsc.fixed')
# for key in data.keys():
#     path = f"data/superglue/wsc/{key}.pkl"
#     # d = load_dataset('glue', dataset)[key]
#     # for x in d:
#     #     print(x)
#     pickle.dump(data[key], open(path, 'wb'))
