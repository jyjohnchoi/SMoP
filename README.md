# SMoP

Code for our paper "SMoP: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts".

To run, download the dataset with 
```
python data/superglue/get_huggingface_superglue.py.
```


Then, run scripts/train.py with sufficient arguments.

## Example run
```
python scripts/train.py --lr 0.5 
                        --batch_size 32 
                        --epoch 50 
                        --max_length 512 
                        --model_name_or_path t5-base 
                        --tokenizer_name_or_path t5-base 
                        --warmup_ratio 0.06 
                        --method prompt-routing 
                        --dataset_name rte_superglue 
                        --num_virtual_tokens 5 
                        --num_virtual_tokens_full 20 
                        --perturb_router True 
                        --topk 1

```
