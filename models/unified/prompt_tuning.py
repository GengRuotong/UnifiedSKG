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

        self.preseqlen = args.prompt_tuning.prompt_sequence_length

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
        from ..prompt.modeling_mt5 import MT5ForConditionalGeneration
        from ..prompt.modeling_t5_prompt import T5ForConditionalGeneration

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
        

        # Prompt related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())
        self.prompt_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.prompt_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.dropout = nn.Dropout(args.prompt_tuning.prompt_dropout)

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        if self.args.model.freeze_prefix:
            for param in self.prompt_dec.parameters():
                param.requires_grad = False
            for param in self.prompt_enc.parameters():
                param.requires_grad = False

    def get_prompt(self, bsz=None, sample_size=1, description=None):
        '''
        past_key_values ​​adjusted to the required format
        past_key_values(tuple(tuple(torch.FloatTensor))of lengthconfig.n_layerswith each tuple having 4 tensors of shape
        (batch_size, num_heads, sequence_length - 1, embed_size_per_head)) — Contains precomputed key and value hidden states 
        of the attention blocks. Can be used to speed up decoding.
        '''
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)

        # prompt decoder
        prompts_dec = self.prompt_dec(input_tokens)
        if description is not None:
            prompts_dec = prompts_dec + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        prompts_dec = self.dropout(prompts_dec)
        bsz, seqlen, _ = prompts_dec.shape

        # prompt encoder
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        prompts_enc = self.prompt_enc(input_tokens_enc)
        if description is not None:
            prompts_enc = prompts_enc + description.unsqueeze(1)
        prompts_enc = self.dropout(prompts_enc)
        bsz_enc, seqlen, _ = prompts_enc.shape

        result = dict()
        result["encoder_prompt"] = {
            "prompts": prompts_enc,
            "inputs_prefix_attention_mask": torch.ones(bsz, seqlen)
            }
        result["decoder_prompt"] = {
            "prompts": prompts_dec,
            "inputs_prefix_attention_mask": torch.ones(bsz_enc, seqlen)
            }

        return result

    def get_description_representation(self, kwargs):
        if self.args.model.use_description and self.args.model.map_description:
            description_input_ids = kwargs.pop("description_input_ids")
            description_attention_mask = kwargs.pop("description_attention_mask")
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                description_outputs = self.pretrain_model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                description_outputs = self.pretrain_model.model.encoder(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                )
                description = description_outputs.last_hidden_state[:, 0]  # TODO: the first token from the encoder.
            else:
                raise ValueError()
        else:
            description = None

        return description

    '''
    The forward function is to directly input a series of obtained parameters into the model 
    after splicing attention_mask and prefix_attention_mask. 
    model source code(e.g. BertSelfAttention.forward()):
        elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    '''
    def forward(self,
                input_ids,
                attention_mask,
                labels,
                **kwargs,
                ):
        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, description=description_representation,
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

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams'], description=description_representation,
        )
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids