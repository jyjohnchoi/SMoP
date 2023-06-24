# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import math
import copy
from dataclasses import dataclass, field
from typing import Optional, Union
from collections import defaultdict

import torch
import torch.nn.functional as F

from transformers import BertForSequenceClassification, BertTokenizer, T5Tokenizer, BertConfig

from ..utils import PeftType, PromptLearningConfig


class PromptRoutingInit(str, enum.Enum):
    TEXT = "TEXT"
    RANDOM = "RANDOM"


@dataclass
class PromptRoutingConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PromptRouting`].
    """
    prompt_routing_init: Union[PromptRoutingInit, str] = field(
        default=PromptRoutingInit.RANDOM,
        metadata={"help": "How to initialize the prompt tuning parameters"},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )
    num_virtual_tokens_full: Optional[int] = field(
        default=100, 
        metadata={
            "help": "The number of target tokens for top-k routing"
        }
    )
    perturb_router: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If true, a random perturbation is added to the routing values"
        }
    )
    topk: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of experts to access per sample"
        }
    )
    stochastic: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If true, no router model is used and inputs are concatenated with random prompts. During inference, the sum of the prompt is used"
        }
    )
    gumbel: Optional[bool] = field(
        default=False, 
        metadata={
            "help": "Whether to use the auxiliary load balancing loss or not."
        }    
    )

    def __post_init__(self):
        self.peft_type = PeftType.PROMPT_ROUTING

       
class PromptRoutingEmbedding(torch.nn.Module):
    def __init__(self, config, word_embeddings):
        super().__init__()
        self.config = config
        total_virtual_tokens = config.num_virtual_tokens_full # * config.num_transformer_submodules
        self.embedding = torch.nn.Embedding(total_virtual_tokens, config.token_dim)
        
        # Linear router
        assert config.num_virtual_tokens_full % config.num_virtual_tokens == 0
        self.n_routes = config.num_virtual_tokens_full // config.num_virtual_tokens
        linear = torch.nn.Linear(config.token_dim, self.n_routes, bias=False)
        
        torch.nn.init.orthogonal_(linear.weight.data)

        if self.config.perturb_router:
            sigma = 1
            self.router = torch.nn.Sequential(
                linear,
                GaussianNoise(sigma=sigma),
                # torch.nn.Dropout(p=0.2)
            )        
        else:
            self.router = torch.nn.Sequential(
                # torch.nn.utils.weight_norm(linear, dim=0),
                linear,
                torch.nn.Dropout(p=0.2),
            )        
        
        self.load_infos = {"Train": defaultdict(list), "Validation": defaultdict(list), "Test": defaultdict(list)}
        self.router.add_module("softmax", torch.nn.Softmax(dim=-1))
        self.load_counts = torch.zeros((self.n_routes,), device='cuda', requires_grad=False)
        self.probs_sum = torch.zeros_like(self.load_counts)
        self.analysis = False


    def forward(self, indices, input_ids, inputs_embeds, attention_mask, base_model=None):
        batch_size = inputs_embeds.shape[0]
        num_virtual_tokens_full = self.config.num_virtual_tokens_full
        num_virtual_tokens = self.config.num_virtual_tokens

        hiddens = inputs_embeds
        sentence_sum = torch.sum(hiddens * attention_mask.unsqueeze(-1), dim=1)
        non_zero_count = torch.clamp(torch.sum(attention_mask, dim=1, keepdim=True), min=1)
        sentence_embeds = sentence_sum / non_zero_count.float()

        if not self.config.stochastic:
            probs = self.router.forward(sentence_embeds)
            if self.config.gumbel:
                probs = gumbel_softmax(probs, k=1, hard=True)
            probs_mean = torch.mean(probs, dim=0)
            load_routes = torch.argmax(probs.detach(), dim=1)
            self.probs_sum = self.probs_sum.detach() + torch.sum(probs.detach(), dim=0)
            self.load_routes = load_routes
            values, idx = torch.topk(probs, k=self.config.topk, dim=-1)
        else:
            idx = torch.randint(low=0, high=self.n_routes, size=(batch_size, 1), device='cuda')
        
        k = idx.shape[1]
        load_counts = torch.bincount(torch.flatten(idx), minlength=self.n_routes) / (batch_size * k)
        self.load_counts = self.load_counts.detach() + load_counts.detach() * batch_size
        if k == 1:
            idx = idx*self.config.num_virtual_tokens + torch.arange(0, self.config.num_virtual_tokens, device='cuda').unsqueeze(0)
        else:
            idx = (idx*self.config.num_virtual_tokens).unsqueeze(-1) + torch.arange(0, self.config.num_virtual_tokens, device='cuda').unsqueeze(0).unsqueeze(0)
        
        if not self.config.stochastic:
            prompt_embeddings = self.embedding(idx) * values.unsqueeze(-1).unsqueeze(-1)
            prompt_embeddings = torch.sum(prompt_embeddings, dim=1).squeeze()
            balancing_factor = probs_mean * load_counts #  probs_mean * load_counts 
        else:
            if self.training:
                prompt_embeddings = self.embedding(idx)
            else:
                prompt_embeddings = torch.mean(torch.stack(torch.chunk(self.embedding.weight.data, self.n_routes, dim=0)), dim=0)
                prompt_embeddings = prompt_embeddings.repeat(batch_size, 1, 1)
        if prompt_embeddings.dim() == 2:
            prompt_embeddings = prompt_embeddings.unsqueeze(0)

        if self.analysis:
            indices = torch.full((batch_size, 1), self.prompt_index * self.config.num_virtual_tokens, device='cuda').long() + torch.arange(0, self.config.num_virtual_tokens, device='cuda').repeat(batch_size, 1)
            prompt_embeddings = self.embedding(indices)# * probs[:, self.prompt_index].unsqueeze(-1).unsqueeze(-1)
                

        return prompt_embeddings

    def save_load_information(self, data_idx, split=None):
        data_idx = data_idx.tolist()
        if split is None:
            split = "Train" if self.training else "Validation"
        for i, index in enumerate(data_idx):
            self.load_infos[split][index].append(self.load_routes[i].item())

    # --- code for token-level analysis --- #
    def activate_analysis(self):
        self.analysis = True

    def disable_analysis(self):
        self.analysis = False

    def fix_prompt(self, index):
        self.prompt_index = index

    def fix_token(self, token_index, prompt_index):
        self.token_index = token_index
        self.prompt_index = prompt_index
    
    def print_and_reset_load_counts(self):
        print(self.load_counts.long())
        self.load_counts.fill_(0)
        self.probs_sum.fill_(0)

    def reset_load_counts(self):
        self.load_counts.fill_(0)
        self.probs_sum.fill_(0)
    
class GaussianNoise(torch.nn.Module):
    """Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.sigma * self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 

class SoftmaxWithTemperature(torch.nn.Module):
    def __init__(self, dim, temperature):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x):
        x = x / self.temperature
        x = torch.nn.functional.softmax(x, dim=self.dim)
        return x

def gumbel_softmax(logits, k, temperature=1.0, hard=False):
    # Add Gumbel noise to the logits
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise))
    noisy_logits = (logits + gumbel) / temperature

    # Apply the Softmax function
    probs = torch.nn.functional.softmax(noisy_logits, dim=-1)

    if hard:
        # Apply a one-hot encoding to the top-k values
        _, top_k = probs.topk(k, dim=-1)
        one_hot = torch.zeros_like(logits).scatter_(-1, top_k, 1.0)
        output = one_hot - probs.detach() + probs
    else:
        # Return a differentiable mixture of one-hot and Softmax
        output = probs

    return output