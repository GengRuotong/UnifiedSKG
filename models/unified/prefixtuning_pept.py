#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoTokenizer
from .tokenizer_chn import T5PegasusTokenizer
from .base import PushToHubFriendlyModel
from ..prompt.modeling_auto import AutoModelForSeq2SeqLM


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim

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
        elif isinstance(self.pretrain_model, T5ForConditionalGeneration):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        elif isinstance(self.pretrain_model, MT5ForConditionalGeneration):
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
     
        self.register_buffer('layer_id', torch.cat([torch.arange(12)+1.0, torch.zeros(self.preseqlen-12)], dim=0).long())
        self.register_buffer('domain_id', torch.cat([torch.arange(5)+1.0, torch.zeros(self.preseqlen-5)], dim=0).long())
        self.register_buffer('position_id', torch.cat([torch.arange(3)+1.0, torch.zeros(self.preseqlen-3)], dim=0).long())
        self.layer_emb = nn.Embedding(self.preseqlen, self.n_embd)
        self.domain_emb = nn.Embedding(self.preseqlen, self.n_embd)
        self.position_emb = nn.Embedding(self.preseqlen, self.n_embd)
        self.fuse= nn.Sequential(
            nn.Linear(3*self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, 64),
        )
        self.project = nn.Linear(64, 3*self.match_n_layer * 2 * self.match_n_head * self.match_n_embd)
        self.dropout = nn.Dropout(args.prefix_tuning.prefix_dropout)

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        
        if self.args.model.freeze_prefix:
            for param in self.layer_emb.parameters():
                param.requires_grad = False
            for param in self.domain_emb.parameters():
                param.requires_grad = False
            for param in self.position_emb.parameters():
                param.requires_grad = False
            for param in self.fuse.parameters():
                param.requires_grad = False
            for param in self.project.parameters():
                param.requires_grad = False

    def get_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None):
        '''
        past_key_values ​​adjusted to the required format
        past_key_values(tuple(tuple(torch.FloatTensor))of lengthconfig.n_layerswith each tuple having 4 tensors of shape
        (batch_size, num_heads, sequence_length - 1, embed_size_per_head)) — Contains precomputed key and value hidden states 
        of the attention blocks. Can be used to speed up decoding.
        '''

        input_layers = self.layer_id.unsqueeze(0).expand(bsz, -1) # bsz, seqlen
        input_domains = self.domain_id.unsqueeze(0).expand(bsz, -1)
        input_positions = self.position_id.unsqueeze(0).expand(bsz, -1)
        temp_layer_control = self.layer_emb(input_layers)
        temp_domain_control = self.domain_emb(input_domains)
        temp_position_control = self.position_emb(input_positions)
        temp_control = torch.cat([temp_layer_control,temp_domain_control,temp_position_control], dim=-1)
        temp_fuse = self.fuse(temp_control)
        past_key_values_total = self.project(temp_fuse)
        past_key_values_total_list = list(past_key_values_total.split(self.match_n_layer * 2 * self.match_n_head * self.match_n_embd, dim=-1))
        past_key_values = past_key_values_total_list[0]
        past_key_values_dec = past_key_values_total_list[1]
        past_key_values_enc = past_key_values_total_list[2]
        
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        ) 
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
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

        return result


    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs,
                ):
        bsz = input_ids.shape[0]


        past_prompt = self.get_prompt(
            bsz=bsz
        )

        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        ).loss
        return {'loss': loss}

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]


        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams'], 
        )
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids