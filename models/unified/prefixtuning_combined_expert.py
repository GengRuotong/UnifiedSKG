#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
from torch import nn
from transformers import AutoTokenizer
from .tokenizer_chn import T5PegasusTokenizer

from .base import PushToHubFriendlyModel
from ..prompt.modeling_auto import AutoModelForSeq2SeqLM
from utils.moe.base_layer import BaseLayer, get_phm_rule_expert, get_phm_rule_shared
from utils.moe.route import get_gate_instance
PROMPT_POS_LIST_W_DE = ["decoder_prompt","cross_attention_prompt","encoder_prompt"]
PROMPT_POS_LIST_WO_DE = ["cross_attention_prompt","encoder_prompt"]

def aggregate_prompt(
    expert_prompt_dict, past_prompt_dict: OrderedDict,  task_names_list=None, num_up_layer=6,
):
    """
    past_prompt_dict: a dict of past_prompt from different tasks.
    """
    constructed_prompt = None
    for task_name, prompt_of_this_task in past_prompt_dict.items():
            if not constructed_prompt:
                constructed_prompt = [{k: {_k: _v for _k, _v in v.items()} for k, v in item.items()} for item in prompt_of_this_task]
                continue
            for layer_number, prompt_of_this_task_in_this_layer in enumerate(
                prompt_of_this_task
            ):
                constructed_prompt_layer = constructed_prompt[layer_number]
                if layer_number < num_up_layer:
                    prompt_pos_list = PROMPT_POS_LIST_W_DE
                else:
                    prompt_pos_list = PROMPT_POS_LIST_WO_DE
                for prompt_pos in prompt_pos_list:
                    for key_value_attention_mask in [
                        "prev_key",
                        "prev_value",
                        "prev_key_padding_mask",
                    ]:
                        if key_value_attention_mask == "prev_key_padding_mask":
                            constructed_prompt_layer[prompt_pos][
                                key_value_attention_mask
                            ] = torch.cat(
                                [
                                    constructed_prompt_layer[prompt_pos][
                                        key_value_attention_mask
                                    ],
                                    prompt_of_this_task[layer_number][prompt_pos][
                                        key_value_attention_mask
                                    ],
                                ],
                                dim=1,
                            )
                        else:
                            constructed_prompt_layer[prompt_pos][
                                key_value_attention_mask
                            ] = torch.cat(
                                [
                                    constructed_prompt_layer[prompt_pos][
                                        key_value_attention_mask
                                    ],
                                    prompt_of_this_task[layer_number][prompt_pos][
                                        key_value_attention_mask
                                    ],
                                ],
                                dim=2,
                            )
    for layer_number, prompt_in_this_layer in enumerate(constructed_prompt):
        if layer_number >= num_up_layer:
            num_base_index = layer_number-num_up_layer
            prompt_in_this_layer["decoder_prompt"] = expert_prompt_dict[num_base_index]["decoder_prompt"]
    return constructed_prompt


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The Multi-task prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim
        self.task_num = args.prefix_tuning.task_num

        # expert
        self.moe_expert_count = args.expert.moe_expert_count
        self.num_base_layers = args.expert.num_base_layers
        self.block_w_base = args.expert.block_w_base
        self.project_struct = args.expert.project_struct
        self.use_xmoe = args.expert.use_xmoe
        self.phm_rule_per_layer_share = args.expert.phm_rule_per_layer_share
        self.share_kv = args.expert.share_kv_down
        self.phm_expert = args.expert.phm_expert
        
        # phm
        self.phm_dim = args.model.phm_dim
        self.phm_rank=args.model.phm_rank
        self.factorized_phm = args.model.factorized_phm
        self.strategy = args.model.strategy
        # combine
        self.load_multiple_prefix_module_weights_from = args.load_multiple_prefix_module_weights_from
        print(self.block_w_base, self.strategy)

        # need to mention, prefix length is the "task_name.split('_')[-1]",
        # which means the name is format as "'name' + '_' + 'prefix length'"
        self.task_name_prefix_len_module_weight_location = [
            (
                "_".join(task_name.split("_")[:-1]),
                int(task_name.split("_")[-1]),
                module_weight_location
            ) for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from
        ]

        total_prelen = 0
        for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from:
            total_prelen += int(task_name.split("_")[-1])
        self.total_prelen = total_prelen

        # Load tokenizer and model.
        if args.bert.description == 't5-pegasus':
            self.tokenizer = T5PegasusTokenizer.from_pretrained(args.bert.location, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
            args.bert.location, use_fast=False
        )
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(args.bert.location)
        self.config = self.pretrain_model.config
        from ..prompt.modeling_bart import BartForConditionalGeneration
        from ..prompt.modeling_t5 import T5ForConditionalGeneration
        from ..prompt.modeling_mt5 import MT5ForConditionalGeneration

        if isinstance(self.pretrain_model, BartForConditionalGeneration):
            self.match_n_layer = self.config.decoder_layers
            self.match_n_head = self.config.decoder_attention_heads
        elif isinstance(
            self.pretrain_model, T5ForConditionalGeneration):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
        elif isinstance(self.pretrain_model, MT5ForConditionalGeneration):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        else:
            raise ValueError("Other models are not supported yet!")

        self.n_embd = self.config.d_model
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        # Prefix related.
        # The Multi prefix modules!
        # The task-prefix modules from all specific tasks
        self.num_up_layers = self.match_n_layer - self.num_base_layers
        self.multi_prefix = nn.ModuleDict(
            {
                task_name: nn.ModuleDict(
                    {   "wte": nn.Embedding(
                            prefix_len, self.n_embd
                        ),
                        "control_trans": nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.ReLU(),
                            nn.Linear(
                                self.mid_dim, self.num_up_layers * 2 * self.n_embd
                            ),
                        ),
                        "norm": nn.LayerNorm(self.match_n_embd),
                        "wte_enc": nn.Embedding(
                            prefix_len, self.n_embd
                        ),
                        "control_trans_enc": nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.ReLU(),
                            nn.Linear(
                                self.mid_dim, self.match_n_layer * 2 * self.n_embd
                            ),
                        ),
                        "norm_enc": nn.LayerNorm(self.match_n_embd),
                        "wte_dec": nn.Embedding(
                            prefix_len, self.n_embd
                        ),
                        "control_trans_dec": nn.Sequential(
                            nn.Linear(self.n_embd, self.mid_dim),
                            nn.ReLU(),
                            nn.Linear(
                                self.mid_dim, self.match_n_layer * 2 * self.n_embd
                            ),
                        ),
                        "norm_dec": nn.LayerNorm(self.match_n_embd),
                        "dropout": nn.Dropout(args.prefix_tuning.prefix_dropout),
                    }
                )
                for task_name, prefix_len, module_weight_location in self.task_name_prefix_len_module_weight_location
            }
        ) 

        if 'decoder' in self.block_w_base:
            self.wte_expert = nn.Embedding(self.total_prelen, self.n_embd)
            # self.wte_expert.weight.data = self.initial_expert_wte()
            self.gate = get_gate_instance(
                model_dim=self.n_embd,
                num_expert=self.moe_expert_count,
                gate_type='Top2Gate',
                base_layer_num=self.num_base_layers,
                use_xmoe=self.use_xmoe
            )

            self.phm_rule_expert_down = get_phm_rule_expert(
                base_layer_num=self.num_base_layers,
                phm_dim=self.phm_dim, 
                phm_rule_expert_share=True,
                strategy=self.strategy,
                share_kv=self.share_kv)

            self.phm_rule_expert_up = get_phm_rule_expert(
                base_layer_num=self.num_base_layers,
                phm_dim=self.phm_dim, 
                phm_rule_expert_share=True,
                strategy=self.strategy)

            self.phm_rule_shared_down = get_phm_rule_shared(
                phm_dim=self.phm_dim,
                moe_expert_count=self.moe_expert_count,
                phm_rule_per_layer_share=self.phm_rule_per_layer_share,
                strategy=self.strategy,
                share_kv=True
            )
            self.phm_rule_shared_up = get_phm_rule_shared(
                phm_dim=self.phm_dim,
                moe_expert_count=self.moe_expert_count,
                phm_rule_per_layer_share=self.phm_rule_per_layer_share,
                strategy=self.strategy
            )

            base_layer_net = [BaseLayer(
                                in_features=self.n_embd,
                                mid_features=self.mid_dim,
                                out_features=2 * self.match_n_head * self.match_n_embd,
                                moe_expert_count=self.moe_expert_count,
                                gate=self.gate[i],
                                project_struct=self.project_struct,
                                phm_rule_expert_down=self.phm_rule_expert_down[i],
                                phm_rule_expert_up=self.phm_rule_expert_up[i],
                                phm_rule_shared_down=self.phm_rule_shared_down,
                                phm_rule_shared_up=self.phm_rule_shared_up,
                                factorized_phm=self.factorized_phm,
                                phm_rank=self.phm_rank,
                                strategy=self.strategy,
                                share_kv=self.share_kv) for i in range(self.num_base_layers)]
            self.base_layer_net = nn.ModuleList(base_layer_net)
            self.dropout = nn.Dropout(args.prefix_tuning.prefix_dropout)
            self.norm = nn.LayerNorm(self.match_n_embd)
        # The shared-prefix module
        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False

        if self.args.model.freeze_task_specific_prefix:
            for param_name, param in self.multi_prefix.named_parameters():
                for (
                    task_name,
                    prefix_len,
                    module_weight_location,
                ) in self.task_name_prefix_len_module_weight_location:
                    if param_name.startswith(task_name):
                        param.requires_grad = False

    def get_prompt_expert(self, bsz, total_prefix_len):
        input_tokens = (
            torch.arange(total_prefix_len)
            .long()
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        input_tokens = (
            input_tokens.to("cuda")
            if torch.cuda.is_available()
            else input_tokens.to("cpu")
        )
        balance_loss = 0.0
        # Decoder Prefix
        temp_control = self.wte_expert(input_tokens)
        if 'decoder' in self.block_w_base:
            base_layer_control_list = []
            for i in range(self.num_base_layers):
                base_layer_control_per_layer, l_aux = self.base_layer_net[i](temp_control)
                balance_loss += l_aux
                base_layer_control_list.append(base_layer_control_per_layer)
            base_layer_control = torch.cat(base_layer_control_list, dim=-1)
            past_key_values = base_layer_control
        res_temp_control = temp_control.repeat(1, 1, 2 * self.num_base_layers)
        past_key_values += res_temp_control

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.num_base_layers * 2, self.match_n_head, self.match_n_embd
        ) 
        past_key_values = self.dropout(past_key_values)
        past_key_values = self.norm(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        result = []
        for _, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            result.append(temp)
        return result, balance_loss

    def get_prompt(self, task_name, prefix_len, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = (
            torch.arange(prefix_len)
            .long()
            .unsqueeze(0)
            .expand(bsz, -1)
        )
        input_tokens = (
            input_tokens.to("cuda")
            if torch.cuda.is_available()
            else input_tokens.to("cpu")
        )
        temp_control = self.multi_prefix[task_name]["wte"](input_tokens)
        past_key_values = self.multi_prefix[task_name]["control_trans"](temp_control)
        res_temp_control = temp_control.repeat(1, 1, 2 * self.num_up_layers)
        past_key_values += res_temp_control
        
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.num_up_layers * 2, self.match_n_head, self.match_n_embd
        ) # bsz = 8, seqlen = 10, past_key_values.shape = torch.Size([8, 10, 24, 12, 64])
        past_key_values = self.multi_prefix[task_name]["dropout"](past_key_values)
        past_key_values = self.multi_prefix[task_name]["norm"](past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.multi_prefix[task_name]["wte_dec"](input_tokens)
        past_key_values_dec = self.multi_prefix[task_name]["control_trans_dec"](
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.multi_prefix[task_name]["dropout"](
            past_key_values_dec
        )
        past_key_values_dec = self.multi_prefix[task_name]["norm_dec"](past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            torch.arange(prefix_len)
            .long()
            .unsqueeze(0)
            .expand(old_bsz, -1)
        )
        input_tokens_enc = (
            input_tokens_enc.to("cuda")
            if torch.cuda.is_available()
            else input_tokens_enc.to("cpu")
        )

        temp_control_enc = self.multi_prefix[task_name]["wte_enc"](input_tokens_enc)
        past_key_values_enc = self.multi_prefix[task_name]["control_trans_enc"](
            temp_control_enc
        )  # bsz, seqlen, layer*emb

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.multi_prefix[task_name]["dropout"](past_key_values_enc)
        past_key_values_enc = self.multi_prefix[task_name]["norm_enc"](past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val_dec in enumerate(past_key_values_dec):
            temp = dict()
            if i < self.num_up_layers:
                key_val = past_key_values_dec[i]
                temp["decoder_prompt"] = {
                    "prev_key": key_val[0].contiguous(),
                    "prev_value": key_val[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool(),
                }
            
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                .to(key_val_dec.device)
                .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                .to(key_val_enc.device)
                .bool(),
            }
            result.append(temp)
        return result

    def initial_expert_wte(self):
        wte_state = []
        for _, module_weight_location in self.load_multiple_prefix_module_weights_from:
            state_dict = torch.load(module_weight_location,  map_location="cpu")
            for weight_name, stored_tensor in state_dict.items():
                if self.phm_expert:
                    if str(weight_name) == 'wte.weight':
                        wte_state.append(stored_tensor)         
        wte_initial = torch.cat(wte_state, dim=0)
        return wte_initial

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        **kwargs,
    ):
        bsz = input_ids.shape[0]

        wte_initial = self.initial_expert_wte()

        # get the past key, value and padding mask of each specific task
        expert_past_prompt, balance_loss = self.get_prompt_expert(
                bsz=bsz,
                total_prefix_len=self.total_prelen
            )

        all_past_prompt = OrderedDict()
        for (
            task_name,
            prefix_len,
            module_weight_location,
        ) in self.task_name_prefix_len_module_weight_location:
            all_past_prompt[task_name] = self.get_prompt(
                bsz=bsz,
                task_name=task_name,
                prefix_len=prefix_len,
            )

        # Task name list, a batch of task name
        task_names_list = [self.args.task_id2task_name[task_id.item()] for task_id in kwargs.pop("task_ids")]

        # do the agg of this prompt(key, value and padding mask)

        past_prompt = aggregate_prompt(
            expert_past_prompt,
            all_past_prompt,
            task_names_list,
            num_up_layer=self.num_up_layers,
        )

        mle_loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        ).loss
        total_loss = mle_loss + 2*balance_loss
        return {"loss": total_loss}

    def generate(self, input_ids, attention_mask, **kwargs):

        bsz = input_ids.shape[0]
        expert_past_prompt, balance_loss = self.get_prompt_expert(
                bsz=bsz,
                total_prefix_len=self.total_prelen
            )

        all_past_prompt = OrderedDict()
        # get the past key, value and padding mask of each specific task
        for (
            task_name,
            prefix_len,
            module_weight_location,
        ) in self.task_name_prefix_len_module_weight_location:
            all_past_prompt[task_name] = self.get_prompt(
                bsz=bsz,
                sample_size=kwargs["num_beams"],
                task_name=task_name,
                prefix_len=prefix_len,
            )
        # Task name list, a batch of task name
        task_names_list = [self.args.task_id2task_name[task_id.item()] for task_id in kwargs.pop("task_ids")]

        # do the agg of this prompt(key, value and padding mask)
        past_prompt = aggregate_prompt(
            expert_past_prompt,
            all_past_prompt,
            task_names_list,
            num_up_layer=self.num_up_layers,
        )

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids