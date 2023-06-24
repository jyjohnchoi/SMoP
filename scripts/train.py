import os
import sys
if "peft" not in os.getcwd():
    os.chdir("peft")
sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import random
from tqdm import tqdm
import argparse
from distutils.util import strtobool as _bool
import time
import re
import string

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from thop import profile

# from torchviz import make_dot, make_dot_from_trace

from tensorboardX import SummaryWriter
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score

from transformers import Trainer, TrainingArguments
from transformers import logging
from transformers import get_linear_schedule_with_warmup
from transformers import Adafactor
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, AutoModelWithLMHead, AutoConfig
from transformers import RobertaForMultipleChoice, T5ForConditionalGeneration

from peft_models import get_peft_config, get_peft_model, PeftConfig, PeftModel, LoraConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig, PromptRoutingConfig, TaskType

from src.t5_with_prefix import T5ForConditionalGenerationWithPrefix
from src.dataset import SuperGlueData
from config.model_config import Config

logging.set_verbosity_error()
# torch.autograd.set_detect_anomaly(True)

# SEQ_2_SEQ_LM -> AutoModelForSeq2SeqLM
# SEQ_CLS -> AutoModelForSequenceClassification

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config()
    # Training arguments 
    try:
        model_name_or_path, tokenizer_name_or_path = cfg.models[args.model_name_or_path], cfg.models[args.tokenizer_name_or_path]
    except KeyError:
        model_name_or_path, tokenizer_name_or_path = args.model_name_or_path, args.tokenizer_name_or_path
    lr = args.lr
    batch_size = args.batch_size
    epoch = args.epoch

    step_batch_size = args.step_batch_size if args.step_batch_size else batch_size
    accum_steps = 1 if step_batch_size is None else batch_size // step_batch_size

    if args.method == "prefix-tuning" or args.method == "full":
        max_length = args.max_length
    else:
        max_length = args.max_length - args.num_virtual_tokens


    # Tensorboard logging
    log_step = 10
    log_dir = f"log/{model_name_or_path}/{args.dataset_name}/{args.method}/"
    os.makedirs(log_dir, exist_ok=True)
    
    if args.method == "prompt-routing":
        if args.stochastic:
            args_string = f"{lr=} {batch_size=} {epoch=} {args.num_virtual_tokens=} {args.num_virtual_tokens_full=} {args.routing_level=} {args.topk=} central_prompt={bool(args.central_prompt)} stochastic=True {args.random_seed=}"
        elif args.gumbel:
            args_string = f"{lr=} {batch_size=} {epoch=} {args.num_virtual_tokens=} {args.num_virtual_tokens_full=} {args.routing_level=} {args.topk=} central_prompt={bool(args.central_prompt)} gumbel=True {args.random_seed=}"
        else:
            args_string = f"{lr=} {batch_size=} {epoch=} {args.num_virtual_tokens=} {args.num_virtual_tokens_full=} perturb_router={bool(args.perturb_router)} {args.routing_level=} {args.topk=} central_prompt={bool(args.central_prompt)} load_balancing={bool(args.load_balancing)} {args.random_seed=}"
    else:
        args_string = f"{lr=} {batch_size=} {epoch=} {args.num_virtual_tokens=} {args.random_seed=}"
    if args.init_text:
        args_string += f" init_text='{args.init_text}'"
    args_string = args_string.replace('args.', "")
    tb_writer = SummaryWriter(os.path.join(log_dir, args_string))
    print(args_string)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_length)
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Dataloader
    text_to_text = 't5' in args.model_name_or_path
    train_dataloader, val_dataloader, test_dataloader, num_labels, metric = prepare_dataset(args.dataset_name, tokenizer, step_batch_size, max_length=max_length, text_to_text=text_to_text, seed=args.random_seed)
    
    ### PEFT configurations
    task_type = "SEQ_CLS" if "roberta" in args.model_name_or_path else "SEQ_2_SEQ_LM"
    # Baseline methods
    if args.method == "lora":
        peft_config = LoraConfig(task_type=task_type, inference_mode=False, r=8, lora_alpha=8, lora_dropout=0.1)
    elif args.method == "prefix-tuning":
        peft_config = PrefixTuningConfig(task_type=task_type, num_virtual_tokens=args.num_virtual_tokens) #, prefix_projection=True, encoder_hidden_size=512)
    elif args.method == "p-tuning":
        peft_config = PromptEncoderConfig(task_type=task_type, num_virtual_tokens=args.num_virtual_tokens, encoder_hidden_size=128)
    elif args.method == "prompt-tuning":
        if args.init_text:
            peft_config = PromptTuningConfig(task_type=task_type, num_virtual_tokens=args.num_virtual_tokens, prompt_tuning_init="TEXT", prompt_tuning_init_text=args.init_text, tokenizer_name_or_path=tokenizer_name_or_path)
        else:
            peft_config = PromptTuningConfig(task_type=task_type, num_virtual_tokens=args.num_virtual_tokens)
    # My methods
    elif args.method == "prompt-routing":
        if args.init_text:
            peft_config = PromptRoutingConfig(task_type=task_type, num_virtual_tokens=args.num_virtual_tokens, prompt_routing_init="TEXT", prompt_routing_init_text=args.init_text, tokenizer_name_or_path=tokenizer_name_or_path, num_virtual_tokens_full=args.num_virtual_tokens_full, perturb_router=args.perturb_router, routing_level=args.routing_level, central_prompt=args.central_prompt, topk=args.topk, stochastic=args.stochastic, load_balancing=args.load_balancing)
        else:
            peft_config = PromptRoutingConfig(task_type=task_type, num_virtual_tokens=args.num_virtual_tokens, num_virtual_tokens_full=args.num_virtual_tokens_full, perturb_router=args.perturb_router, routing_level=args.routing_level, central_prompt=args.central_prompt, topk=args.topk, stochastic=args.stochastic, gumbel=args.gumbel)

    # Pre-trained model configuraitons
    config = AutoConfig.from_pretrained(model_name_or_path)
    if config.pad_token_id == None: 
        config.pad_token = tokenizer.pad_token
        config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = num_labels

    # load model
    # RoBERTa
    if 'roberta' in args.model_name_or_path:
        if args.dataset_name == "copa":
            model = RobertaForMultipleChoice.from_pretrained(model_name_or_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)    
    # T5
    elif 't5' in args.model_name_or_path:
        # if args.method == 'prefix-tuning':
        #     model = T5ForConditionalGenerationWithPrefix.from_pretrained(model_name_or_path, config=config)
        # else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    model.resize_token_embeddings(len(tokenizer))
    if args.method != "full":
        model = get_peft_model(model, peft_config)
        # model = FunctionalPromptModelForSeq2SeqLM(model, peft_config)
        model.print_trainable_parameters()
    model = model.to(device)

    # model.prompt_encoder.router[0].weight.data = kmean_cluster(args.random_seed)

    tuning_params = [p for n, p in list(model.named_parameters()) if p.requires_grad]
    
    # Optimizer, Scheduler
    total_steps = len(train_dataloader) // accum_steps * epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    if "t5" in args.model_name_or_path:
        if args.method == 'prompt-routing':
            tuning_params = [
                # {'params': model.base_model.parameters(), 'lr': args.lr},
                {'params': model.prompt_encoder.embedding.parameters(), 'lr': args.lr}, 
                {'params': model.prompt_encoder.router.parameters(), 'lr': args.lr}
                ]
            optimizer = Adafactor(tuning_params, lr=args.lr, relative_step=False, weight_decay=1e-5, scale_parameter=False)
        else:    
            optimizer = Adafactor(tuning_params, lr=args.lr, relative_step=False, weight_decay=1e-5, scale_parameter=False)
    else:
        optimizer = torch.optim.AdamW(tuning_params, lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler()
    step = 0
    log_loss = []
    
    # Early stopping
    max_patience = 10

    patience_loss = max_patience
    min_val_loss = 100
    max_test_result_loss = 0
    saved_epoch_loss = 0

    patience_score = max_patience
    max_val_score = 0
    max_test_result_score = 0
    saved_epoch_score = 0

    train_time_total = 0

    start_time = time.time()

    mse = torch.nn.MSELoss()

    log_path = f"{log_dir}/{args_string}/load_counts_log.txt"
    l = open(log_path, 'wt')


    for e in range(epoch):
        l.write(f"Epoch {e+1}\n")
        model.train()
        train_loss, step, log_loss, train_time = train_epoch(model, optimizer, scheduler, scaler, train_dataloader, accum_steps, tb_writer, step, log_step, log_loss, device, metric, tokenizer)
        train_time_total += train_time
        if args.method == 'prompt-routing':
            l.write(f"Train: {model.prompt_encoder.load_counts}\n")
            model.prompt_encoder.print_and_reset_load_counts() 
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, preds, answers = evaluate_epoch(model, scaler, val_dataloader, device, tokenizer, text_to_text, test=False)
            if args.method == 'prompt-routing':
                l.write(f"Validation: {model.prompt_encoder.load_counts}\n")
                model.prompt_encoder.print_and_reset_load_counts() 
            # n = preds.count('entailment')
            # print(n, len(preds) - n)
        val_score = score(metric, preds, answers, is_wsc=False)#=(args.dataset_name=='wsc'))
        print("Epoch {} | Train loss: {:.5f} | Validation loss: {:.5f} | Validation acc: {:.1f}".format(e+1, train_loss, val_loss, val_score*100))
        is_better_loss = (val_loss < min_val_loss)
        is_better_score = (val_score > max_val_score)

        # Test
        # print("If only validation set, result would be {:.1f}".format((val_score * 276 + test_score * 278)/554*100))

        # Log results to tensorboard
        tb_writer.add_scalar('validation/loss', val_loss, e+1)
        tb_writer.add_scalar('validation/performance', val_score, e+1)
        # tb_writer.add_scalar('test/performance', test_score, e+1)

        # Early stoppings
        os.makedirs(f"models/{args.model_name_or_path}/{args.method}/{args.dataset_name}/", exist_ok=True)

        if is_better_loss and patience_loss > 0:
            print("Updating result due to lower validation loss")
            with torch.no_grad():
                _, preds, answers = evaluate_epoch(model, scaler, test_dataloader, device, tokenizer, text_to_text, test=True)
            test_score = score(metric, preds, answers, is_wsc=(args.dataset_name=='wsc'))
            min_val_loss = val_loss
            max_test_result_loss = test_score
            patience_loss = max_patience
            if args.method != "full":
                peft_model_id = f"models/{args.model_name_or_path}/{args.method}/{args.dataset_name}/{args_string}_early_stop_loss"
                # if args.method == "prompt-routing":
                #     torch.save(model.prompt_encoder, peft_model_id)
                # else:
                #     model.save_pretrained(peft_model_id)             
        else:
            patience_loss -= 1
            if patience_loss == 0:
                print("Patience of loss-based early stopping has reached 0. No more updates.")

        if is_better_score and patience_score > 0:
            # with torch.no_grad():
            #     _, preds, answers = evaluate_epoch(model, scaler, test_dataloader, device, tokenizer, text_to_text, test=True)
            # test_score = score(metric, preds, answers, is_wsc=(args.dataset_name=='wsc'))
            print("Updating result due to higher validation score")

            max_val_score = val_score
            try:
                max_test_result_score = test_score
            except UnboundLocalError:
                with torch.no_grad():
                    _, preds, answers = evaluate_epoch(model, scaler, test_dataloader, device, tokenizer, text_to_text, test=True)
                test_score = score(metric, preds, answers, is_wsc=(args.dataset_name=='wsc'))                
            print("Epoch {} test result: {:.1f}".format(e+1, test_score * 100))
            patience_score = max_patience
            if args.method != "full":
                peft_model_id = f"models/{args.model_name_or_path}/{args.method}/{args.dataset_name}/{args_string}_early_stop_score"
                # if args.method == "prompt-routing":
                #     torch.save(model.prompt_encoder, peft_model_id)
                # else:
                #     model.save_pretrained(peft_model_id)             
        else:
            patience_score -= 1    
            if patience_score == 0:
                print("Patience of score-based early stopping has reached 0. No more updates.")

        if args.method == 'prompt-routing':
            l.write(f"Test: {model.prompt_encoder.load_counts}\n")
            model.prompt_encoder.print_and_reset_load_counts() 
            print()
        
        if patience_score <= 0 and patience_loss <= 0:
            break

    running_time = time.time() - start_time
    avg_time = train_time_total / (e+1)
    # Overall results and saving modules
    print(args_string)
    print("Test result | Loss : {:.1f} | Score : {:.1f}".format(max_test_result_loss * 100, max_test_result_score * 100))
    print("Took {:.2f} minutes for training".format(running_time / 60))
    print("Took {:.2f} seconds per epoch".format(avg_time))

    # if args.epoch > 10:   
    export_dir = f"results/{model_name_or_path}/{args.dataset_name}"
    # else:
    #     export_dir = f"results/{model_name_or_path}/time_measure"
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, f"{args.method}.txt"), 'a') as g:
        output_args = f"{args.method=} {args.dataset_name=} {args_string}".replace("args.", "")            
        g.write(output_args + '\n')
        g.write("Took {:.2f} minutes for training / {:.2f} seconds per epoch".format(running_time / 60, avg_time) + '\n')
        g.write("Test result | Loss : {:.1f} | Score : {:.1f}".format(max_test_result_loss * 100, max_test_result_score * 100) + '\n')

    l.close()

def train_epoch(model, optimizer, scheduler, scaler, train_dataloader, accum_steps, tb_writer, step, log_step, log_loss, device, metric, tokenizer):
    st = time.time()
    model.train()
    criterion = torch.nn.MSELoss()
    epoch_loss = []
    a = 0
    min_len = 512
    # dictionary (key: data index, value: list of routing results)
    for train_sample in tqdm(train_dataloader, ncols=100, desc=f"Train", leave=False):

        ids = train_sample["ids"]
        input_ids = train_sample["input_ids"].to(device)
        # flops, params = profile(model, inputs=(input_ids,))
        # print(flops)
        # print(params)
        if input_ids.shape[1] < min_len:
            min_len = input_ids.shape[1]
        att_mask = torch.ones_like(input_ids) * (input_ids != tokenizer.pad_token_id).long()
        labels = train_sample["label"].to(device)
        labels = labels.float() if metric == "spearman" else labels.long()

    
        
        outputs = model.forward(input_ids=input_ids, attention_mask=att_mask, labels=labels)
        try:
            model.prompt_encoder.save_load_information(ids)
        except:
            pass
        with autocast():
            lm_loss = outputs['loss']

        if args.method == "prompt-routing" and args.stochastic and args.consistency_alpha > 0:
            additional_outputs = model.forward(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            lm_loss_2 = additional_outputs['loss']
            logit1, logit2 = outputs['logits'], additional_outputs['logits']
            if lm_loss == lm_loss_2:
                loss = lm_loss
            else:
                consistency_loss = symmetric_KL_loss(logit1, logit2)
                loss = lm_loss + lm_loss_2 + args.consistency_alpha * consistency_loss
        else:
            try:
                balance_loss = model.prompt_encoder.balancing_loss
                # print(balance_loss)
                if balance_loss > 0:
                    loss = lm_loss + balance_loss
            except AttributeError:
                loss = lm_loss

        scaler.scale(loss).backward()

        if torch.isfinite(loss):
            loss_val = loss.item()
            epoch_loss.append(loss_val)
            log_loss.append(loss_val)
        a += 1
        if a % accum_steps == 0 or a == len(train_dataloader):
            scaler.step(optimizer)
            scheduler.step()       
            scaler.update()
            step += 1
            a = 0
        else:
            continue
        # print(optimizer.param_groups[0]['lr'])
        # print(optimizer.param_groups[1]['lr'])

        optimizer.zero_grad()

        if step % log_step == 0:
            log_loss = sum(log_loss) / len(log_loss)
            # record the average of step losses for smoothness and clarity
            tb_writer.add_scalar('train/train_loss', log_loss/log_step, step//10)
            tb_writer.add_scalar('train/train_lr', optimizer.param_groups[0]['lr'], step//10)
            log_loss = []
            try:
                tb_writer.add_scalar('train/balance_loss', balance_loss, step//10)
                tb_writer.add_scalar('train/lm_loss', lm_loss, step//10)
            except:
                pass
    # print(min_len)
    train_time = time.time() - st
    print("{:.2f} seconds".format(time.time()-st))
    return sum(epoch_loss) / len(epoch_loss), step, log_loss, train_time

def evaluate_epoch(model, scaler, val_dataloader, device, tokenizer, text_to_text, test):
    val_loss = []
    losses = []
    preds, answers = [], []
    with torch.no_grad():
        model.eval()
        for val_sample in tqdm(val_dataloader, ncols=100, desc="Test" if test else "Validation", leave=False):
            ids = val_sample["ids"]
            batch_size = ids.shape[0]
            input_ids = val_sample["input_ids"].to(device)
            labels = val_sample["label"].to(device)
            att_mask = torch.ones_like(input_ids) * (input_ids != tokenizer.pad_token_id).long()
            outputs = model(input_ids=input_ids, attention_mask=att_mask, labels=labels, output_attentions=True)

            if text_to_text:
                pred = model.generate(input_ids=input_ids, attention_mask=att_mask, max_new_tokens=10)
            else:
                pred = outputs["logits"]
                if pred.shape[1] > 1:
                    pred = torch.argmax(pred, dim=-1)
                if pred.shape[0] > 1: # shape가 [1]일 때 squeeze하면 int 되는거 방지
                    pred = pred.squeeze()
                    labels = labels.squeeze()
            if not test:
                try:
                    model.prompt_encoder.save_load_information(ids)
                except:
                    pass

            loss = outputs.loss
            val_loss.append(loss.item())
            
            if text_to_text:
                preds.extend(tokenizer.batch_decode(pred, skip_special_tokens=True))
                labels[labels==-100] = tokenizer.pad_token_id
                answers.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
            else:
                preds.extend(pred.tolist())
                answers.extend(labels.tolist())
    if text_to_text:
        preds = [normalize_answer(p) for p in preds]
        answers = [normalize_answer(a) for a in answers]
        # print(preds)
        # print(answers)
    return sum(val_loss)/ len(val_loss), preds, answers


def symmetric_KL_loss(input, target, reduction='batchmean'):
    """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                    F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
           F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32),
                    F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
    return 0.5 * loss.sum()


def score(metric, preds, answers, is_wsc=False):
    if metric == 'spearman':
        result = spearmanr(preds, answers)[0]
        # print("Spearman correlation: {:.1f}".format(result*100))
        
    elif metric == 'matthews':
        result = matthews_corrcoef(answers, preds)
        # print("Matthews correlation: {:.1f}".format(result*100))

    elif metric == 'acc':
        if is_wsc:
            correct = sum(b in a for a, b in zip(preds, answers))
        else:
            correct = sum(a == b for a, b in zip(preds, answers))
        correct_zeros = sum(a == b for a, b in zip(preds, answers) if b==0)
        correct_ones = sum(a == b for a, b in zip(preds, answers) if b==1)

        result = correct / len(preds)
        # print("Accuracy: {:.1f}".format(result * 100))

    elif metric == 'f1':       
        result = f1_score(answers, preds, average='macro')
        # print("F1 Score: {:.1f}".format(result * 100))

    elif metric == 'f1a':
        result = f1_score(answers, preds, average="macro")
        # print("F1a Score: {:.1f}".format(result * 100))

    return result


def prepare_dataset(dataset_name, tokenizer, batch_size, max_length, text_to_text, seed):
    superglue_dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'rte_superglue', 'wic', 'semeval']
    # assert dataset_name in dataset_names
    dataset_metrics = {'multirc': 'f1a'}
    try:
        metric = dataset_metrics[dataset_name]
    except KeyError:
        metric = 'acc'
    
    if dataset_name in superglue_dataset_names:
        train_dataset = SuperGlueData(dataset_name, 'train', tokenizer, max_length=max_length, text_to_text=text_to_text)
        val_dataset = SuperGlueData(dataset_name, 'dev', tokenizer, max_length=max_length, text_to_text=text_to_text)
        test_dataset = SuperGlueData(dataset_name, 'test', tokenizer, max_length=max_length, text_to_text=text_to_text)
    else:
        raise ValueError("Wrong dataset name. Try again.")
    
    train_dataset.create_dataset()
    val_dataset.create_dataset()
    test_dataset.create_dataset()
    num_labels = train_dataset.num_labels

    def collate_fn(examples):
        ids = [example["ids"] for example in examples]
        input_ids = [example["input_ids"] for example in examples]
        segment_ids = [example["segment_ids"] for example in examples]
        labels = [example["label"] for example in examples]

        if input_ids[0].ndim == 1:
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True)
        else: # multiple choice
            input_ids = [d.transpose(0, 1) for d in input_ids]
            segment_ids = [d.transpose(0, 1) for d in segment_ids]   
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True)
            padded_input_ids = padded_input_ids.transpose(1, 2)
            padded_segment_ids = padded_segment_ids.transpose(1, 2)

        if type(labels[0]) == int: # classfication
            padded_labels = torch.LongTensor(labels)
        else: # text_to_text
            labels = [torch.LongTensor(d) for d in labels]
            padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"ids": torch.LongTensor(ids).contiguous(),
                "input_ids": padded_input_ids.contiguous(),
                "segment_ids": padded_segment_ids.contiguous(),
                "label": padded_labels.contiguous(),
                }
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=4, worker_init_fn=seed_worker, generator=g)   
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=4, worker_init_fn=seed_worker, generator=g)   
    return train_dataloader, val_dataloader, test_dataloader, num_labels, metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=512)

    parser.add_argument('--weight_decay', type=float, default=0) 
    parser.add_argument('--dataset_name', type=str, default='cola')
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--method', type=str, choices=['full', 'lora', 'prefix-tuning', 'p-tuning', 'prompt-tuning', 'prompt-routing']) 
    parser.add_argument('--num_virtual_tokens', type=int, default=None)

    parser.add_argument('--init_text', type=str, default=None)

    parser.add_argument('--num_virtual_tokens_full', type=int, default=None)
    parser.add_argument('--routing_level', type=str, default='prompt')

    # mixture of prompts configs
    parser.add_argument('--consistency_alpha', type=float, default=0.1)
    parser.add_argument('--init_copy', type=_bool, default=True)

    # prompt routing configs
    parser.add_argument('--perturb_router', type=_bool, default=False)
    parser.add_argument('--central_prompt', type=_bool, default=False)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--stochastic', type=_bool, default=False)
    parser.add_argument('--load_balancing', type=_bool, default=True)
    parser.add_argument('--gumbel', type=_bool, default=False)
    args = parser.parse_args()

    train(args)

