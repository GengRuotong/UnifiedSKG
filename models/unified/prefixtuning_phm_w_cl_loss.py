#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformers import AutoTokenizer
from .tokenizer_chn import T5PegasusTokenizer
from .base import PushToHubFriendlyModel
from ..prompt.modeling_auto import AutoModelForSeq2SeqLM
from ..hypercomplex.layers_w_phm_list import PHMLinearBlock
from utils.loss_function.cl_loss_fuction import contrastive_loss

class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.prefix_tuning.prefix_sequence_length
        self.mid_dim = args.prefix_tuning.mid_dim
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
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            PHMLinearBlock(
                in_features=self.mid_dim,
                out_features=self.match_n_head * self.match_n_embd,
                layer_num=self.match_n_layer,
                phm_dim=self.phm_dim,
                factorized_phm=self.factorized_phm
                ),
        )
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                PHMLinearBlock(
                    in_features=self.mid_dim,
                    out_features=self.match_n_head * self.match_n_embd,
                    layer_num=self.match_n_layer,
                    phm_dim=self.phm_dim,
                    factorized_phm=self.factorized_phm
                ),
            )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            PHMLinearBlock(
                in_features=self.mid_dim,
                out_features=self.match_n_head * self.match_n_embd,
                layer_num=self.match_n_layer,
                phm_dim=self.phm_dim,
                factorized_phm=self.factorized_phm
                ),
            )
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans_enc = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                PHMLinearBlock(
                    in_features=self.mid_dim,
                    out_features=self.match_n_head * self.match_n_embd,
                    layer_num=self.match_n_layer,
                    phm_dim=self.phm_dim,
                    factorized_phm=self.factorized_phm
                ),
            )

        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            PHMLinearBlock(
                in_features=self.mid_dim,
                out_features=self.match_n_head * self.match_n_embd,
                layer_num=self.match_n_layer,
                phm_dim=self.phm_dim,
                factorized_phm=self.factorized_phm
                ),
            )

        # Knowledge prompt.
        if self.args.model.knowledge_usage == 'separate':
            self.knowledge_trans_dec = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                PHMLinearBlock(
                    in_features=self.mid_dim,
                    out_features=self.match_n_head * self.match_n_embd,
                    layer_num=self.match_n_layer,
                    phm_dim=self.phm_dim,
                    factorized_phm=self.factorized_phm
                ),
            )

        self.dropout = nn.Dropout(args.prefix_tuning.prefix_dropout)

        if self.args.model.freeze_plm:
            for param in self.pretrain_model.parameters():
                param.requires_grad = False
        if self.args.model.freeze_prefix:
            for param in self.wte.parameters():
                param.requires_grad = False
            for param in self.control_trans.parameters():
                param.requires_grad = False
            for param in self.wte_dec.parameters():
                param.requires_grad = False
            for param in self.control_trans_dec.parameters():
                param.requires_grad = False
            for param in self.wte_enc.parameters():
                param.requires_grad = False
            for param in self.control_trans_enc.parameters():
                param.requires_grad = False

    def get_prompt(self, bsz=None, sample_size=1, description=None, knowledge=None):
        '''
        past_key_values ​​adjusted to the required format
        past_key_values(tuple(tuple(torch.FloatTensor))of lengthconfig.n_layerswith each tuple having 4 tensors of shape
        (batch_size, num_heads, sequence_length - 1, embed_size_per_head)) — Contains precomputed key and value hidden states 
        of the attention blocks. Can be used to speed up decoding.
        '''
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)
        if description is not None:
            temp_control = temp_control + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values = self.control_trans(temp_control)  # [(bsz, seqlen, layer*emb)]

        for i in range(len(past_key_values)):
            if knowledge is not None:
                past_key_values[i] = torch.cat([past_key_values[i], self.knowledge_trans(knowledge.repeat_interleave(sample_size, dim=0))[i]], dim=1)

            bsz, seqlen, _ = past_key_values[i].shape
            past_key_values[i] = past_key_values[i].view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
            )
            past_key_values[i] = self.dropout(past_key_values[i])
            past_key_values[i] = past_key_values[i].permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        if description is not None:
            temp_control_dec = temp_control_dec + description.repeat_interleave(sample_size, dim=0).unsqueeze(1)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # [(bsz, seqlen, layer*emb)]
       
        for i in range(len(past_key_values_dec)):
            if knowledge is not None:
                past_key_values_dec[i] = torch.cat([past_key_values_dec[i], self.knowledge_trans_dec(knowledge.repeat_interleave(sample_size, dim=0))[i]], dim=1)

            bsz, seqlen, _ = past_key_values_dec[i].shape
            past_key_values_dec[i] = past_key_values_dec[i].view(
                bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
            )
            past_key_values_dec[i] = self.dropout(past_key_values_dec[i])
            past_key_values_dec[i] = past_key_values_dec[i].permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)
        if description is not None:
            temp_control_enc = temp_control_enc + description.unsqueeze(1)
        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb
        
        for i in range(len(past_key_values_enc)):
            if knowledge is not None:
                past_key_values_enc[i] = torch.cat([past_key_values_enc[i], self.knowledge_trans_enc(knowledge)[i]], dim=1)

            bsz_enc, seqlen, _ = past_key_values_enc[i].shape
            past_key_values_enc[i] = past_key_values_enc[i].view(
                bsz_enc,
                seqlen,
                self.match_n_layer * 2,
                self.match_n_head,
                self.match_n_embd,
            )
            past_key_values_enc[i] = self.dropout(past_key_values_enc[i])
            past_key_values_enc[i] = past_key_values_enc[i].permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for j in range(len(past_key_values)):
            result.append([])
            for i, key_val in enumerate(past_key_values[j]):
                temp = dict()
                temp["decoder_prompt"] = {
                    "prev_key": key_val[0].contiguous(),
                    "prev_value": key_val[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                        .to(key_val.device)
                        .bool()
                    # bsz, preseqlen
                }
                key_val_dec = past_key_values_dec[j][i]
                temp["cross_attention_prompt"] = {
                    "prev_key": key_val_dec[0].contiguous(),
                    "prev_value": key_val_dec[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                        .to(key_val_dec.device)
                        .bool(),
                }
                key_val_enc = past_key_values_enc[j][i]
                temp["encoder_prompt"] = {
                    "prev_key": key_val_enc[0].contiguous(),
                    "prev_value": key_val_enc[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                        .to(key_val_enc.device)
                        .bool(),
                }
                result[-1].append(temp)
        # result shape is [[dict(), ...dict()], ...[dict(),...,dict()]]
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

    def get_knowledge_representation(self, kwargs):
        if self.args.model.knowledge_usage == 'separate':
            knowledge_input_ids = kwargs.pop("knowledge_input_ids", None)
            knowledge_attention_mask = kwargs.pop("knowledge_attention_mask", None)
            if self.args.bert.location in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]:
                knowledge_outputs = self.pretrain_model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            elif self.args.bert.location in ["facebook/bart-base", "facebook/bart-large"]:
                knowledge_outputs = self.pretrain_model.model.encoder(
                    input_ids=knowledge_input_ids,
                    attention_mask=knowledge_attention_mask,
                )
                knowledge = knowledge_outputs.last_hidden_state
            else:
                raise ValueError()
        elif self.args.model.knowledge_usage == 'concatenate':
            knowledge = None
        else:
            raise ValueError()

        return knowledge

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
        
        # Encode knowledge.
        knowledge_representation = self.get_knowledge_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, description=description_representation, knowledge=knowledge_representation,
        )
        perspect_num = len(past_prompt)
        perspect_hidden_states = []
        for i in range(perspect_num):
            output = self.pretrain_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_prompt=past_prompt[i],
            )
            # odict_keys(['loss', 'logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state'])
            if i == perspect_num - 1:
                mle_loss = output.loss
            else:
                last_hidden_state = output.decoder_hidden_states[-1]
                norm_rep = last_hidden_state / last_hidden_state.norm(dim=2, keepdim=True)
                perspect_hidden_states.append(norm_rep)
        perspect_tensor = torch.stack(perspect_hidden_states, dim=2)  # bsz, sqlen, perspect_num, d_model
    
        cl_loss = contrastive_loss(margin=0.5, perspect_tensor=perspect_tensor, input_ids=input_ids, pad_token_id=0)
        cl_loss = 10 * cl_loss
        print(cl_loss)
        print(mle_loss)
        return {'loss': mle_loss + cl_loss}

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        # Encode description.
        description_representation = self.get_description_representation(kwargs)

        # Encode knowledge.
        knowledge_representation = self.get_knowledge_representation(kwargs)

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams'], description=description_representation, knowledge=knowledge_representation,
        )
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt[-1],
            use_cache=True,
            **kwargs,
        )

        return generated_ids