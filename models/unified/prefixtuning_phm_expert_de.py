#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoTokenizer
from .tokenizer_chn import T5PegasusTokenizer
from .base import PushToHubFriendlyModel
from ..prompt.modeling_auto import AutoModelForSeq2SeqLM
# from utils.moe.base_layer import BaseLayer
from utils.moe.base_layer_w_xmoe import BaseLayer

SUPPORT_PRETRAINED_MODEL = ['BartForConditionalGeneration', 'T5ForConditionalGeneration', 'MT5ForConditionalGeneration']

class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim
        # expert
        self.expert_struct = args.expert.expert_struct
        self.moe_expert_count = args.expert.moe_expert_count
        self.num_base_layers = args.expert.num_base_layers
        # phm
        self.phm_dim = args.model.phm_dim
        self.factorized_phm = args.model.factorized_phm
       
        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        if args.bert.description == 't5-pegasus':
            self.tokenizer = T5PegasusTokenizer.from_pretrained(args.bert.location, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location,
            from_tf=bool(".ckpt" in args.bert.location)
        )
        self.config = self.pretrain_model.config

        from ..prompt.modeling_bart import BartForConditionalGeneration
        from ..prompt.modeling_t5 import T5ForConditionalGeneration
        from ..prompt.modeling_mt5 import MT5ForConditionalGeneration
        if isinstance(self.pretrain_model, BartForConditionalGeneration):
            self.match_n_layer = self.config.decoder_layers
            self.match_n_head = self.config.decoder_attention_heads
            self.n_embd = self.config.d_model
            assert self.n_embd % self.match_n_head == 0
            self.match_n_embd = self.n_embd // self.match_n_head # huggingface BART's dim of kv need to be calculated
        elif isinstance(self.pretrain_model, (T5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        elif isinstance(self.pretrain_model, (MT5ForConditionalGeneration)):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        else:
            raise ValueError("Other models are not supported yet!")

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))
        

        # Prefix related.
        # self.n_embd, self.mid_dim = 768, 512
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())
        self.num_up_layers = self.match_n_layer - self.num_base_layers
        # dec
        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        
        self.down_project = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.ReLU()
            )
        if self.num_up_layers > 0:
            self.up_project = nn.Linear(self.mid_dim, 2*self.num_up_layers*self.match_n_head*self.match_n_embd)
        if 'split_to_layers' in self.expert_struct:
            # out_features = out_features*2*base_layer_num
            self.base_layer = BaseLayer(
                in_features=self.n_embd,
                mid_features=self.mid_dim,
                out_features=self.match_n_head * self.match_n_embd,
                moe_expert_count=self.moe_expert_count,
                base_layer_num=self.num_base_layers,
                phm_dim=self.phm_dim,
                expert_struct=self.expert_struct,
                phm_expert=True
                )
        else:
            raise ValueError("Other expert_structs are not supported yet!")
        self.norm = nn.LayerNorm(self.match_n_embd)
        
        # enc
        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
            )
        self.norm_enc = nn.LayerNorm(self.match_n_embd)

        # cross
        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.ReLU(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        self.norm_dec = nn.LayerNorm(self.match_n_embd)

        self.dropout = nn.Dropout(args.prefix_tuning.prefix_dropout)

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        
        if self.args.model.freeze_prefix:
            for name, module in self.named_modules():
                if 'pretrain' not in name:
                # print(name)
                    for param in module.parameters():
                        param.requires_grad = False
    

    def get_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None):
        '''
        past_key_values ​​adjusted to the required format
        past_key_values(tuple(tuple(torch.FloatTensor))of lengthconfig.n_layerswith each tuple having 4 tensors of shape
        (batch_size, num_heads, sequence_length - 1, embed_size_per_head)) — Contains precomputed key and value hidden states 
        of the attention blocks. Can be used to speed up decoding.
        '''
        balance_loss = 0.0
        # Decoder Prefix
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1) # bsz, seqlen
        temp_control = self.wte(input_tokens)
        down_control = self.down_project(temp_control)
        if self.expert_struct == 'MLP_split_to_layers_w_share':
            base_layer_control, l_aux = self.base_layer(down_control) 
            balance_loss += l_aux
        if self.num_up_layers > 0:
            up_control = self.up_project(down_control)
            base_layer_control = torch.cat([up_control, base_layer_control], dim=-1)
        past_key_values = base_layer_control
        res_temp_control = temp_control.repeat(1, 1, 2 * self.match_n_layer)
        past_key_values += res_temp_control
        
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        ) # bsz = 8, seqlen = 10, past_key_values.shape = torch.Size([8, 10, 24, 12, 64])
        past_key_values = self.dropout(past_key_values)
        past_key_values = self.norm(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(temp_control_dec)
        res_temp_control_dec = temp_control_dec.repeat(1, 1, 2*self.match_n_layer)
        past_key_values_dec += res_temp_control_dec

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = self.norm_dec(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (self.input_tokens.unsqueeze(0).expand(old_bsz, -1))
        temp_control_enc = self.wte_enc(input_tokens_enc)
        past_key_values_enc = self.control_trans_enc(temp_control_enc)
        
        res_temp_control_enc = temp_control_enc.repeat(1, 1, 2*self.match_n_layer)
        past_key_values_enc += res_temp_control_enc
        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = self.norm_enc(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
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

        return result, balance_loss


    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        past_prompt, balance_loss = self.get_prompt(bsz=bsz)

        mle_loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        ).loss
        total_loss = mle_loss + 2*balance_loss
        return {'loss': total_loss}

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        past_prompt, _ = self.get_prompt(bsz=bsz, sample_size=kwargs['num_beams'])
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids


