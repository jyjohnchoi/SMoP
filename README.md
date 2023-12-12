# SMoP (Sparse Mixture-of-Prompts)
 
The repository contains the code for our **EMNLP 2023** paper **"SMoP: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts"**, written by **Joon-Young Choi**, Junho Kim, Jun-Hyung Park, Mok-Wing Lam, and SangKeun Lee.

You can find our paper at https://aclanthology.org/2023.emnlp-main.884/

The codebase will be updated for better use as soon as possible.

## Downloading the Dataset
To run, download the SuperGLUE datasets by
```
python data/superglue/get_huggingface_superglue.py
```
or use your custom dataset. In that case, you need to create your custom `Dataset` class for your dataset in `src/dataset.py` and apply mandatory changes such as importing your dataset or modifying the training script.

## Training
Then, you can execute `scripts/train.py` with training arguments as follows

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

The training script includes the script for evaluation and exporting the results to `results/{model_name_or_path}/{args.dataset_name}/{args.method}.txt` file. 

### Arguments
- `method`: The training method
  - `full`: Full model fine-tuning
  - `prompt-tuning`: Directly fine-tuning the soft prompts (from [Lester et al., 2021](https://aclanthology.org/2021.emnlp-main.243/))
  - `p-tuning`: Utilizing a reparameterization model on the soft prompts (from [Liu et al, 2021](https://arxiv.org/abs/2103.10385))
  - `prompt-routing`: Use **SMoP** for training
 
- `num_virtual_tokens`: The number of the soft prompt tokens attached to the input instance. No impact when the training method is `full`
- `num_virtual_tokens_full`: The total number of soft prompt tokens used during training. For `prompt-routing`, this is different from 'num_virtual_tokens', while it is the same on other methods.
  - For example, if you want to use **SMoP** with 4 soft prompts of length 5, you need to set `num_virtual_tokens` as 5 and `num_virtual_tokens_full` as 20.
 
- `perturb_router`: If True, scaled Gaussian noise (Section 2.3 of our paper) is applied during training.

- `topk`: Number of soft prompt tokens to route each input instance. If larger than 2, the weighted sum of multiple soft prompts is applied.


## Citation
```
@inproceedings{choi-etal-2023-smop,
    title = "{SM}o{P}: Towards Efficient and Effective Prompt Tuning with Sparse Mixture-of-Prompts",
    author = "Choi, Joon-Young  and
      Kim, Junho  and
      Park, Jun-Hyung  and
      Mok, Wing-Lam  and
      Lee, SangKeun",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.884",
    pages = "14306--14316",
}
```

## Acknowledgments
Our implementation is largely based on the [HuggingFace PEFT](https://github.com/huggingface/peft) library.

## Issues
If you have any issues with our paper or the codebase, please leave an issue in the repository or send an email to johnjames@korea.ac.kr
