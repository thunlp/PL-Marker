# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""PyTorch ALBERT model. """

import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_albert import AlbertConfig
from transformers.modeling_bert import ACT2FN, BertEmbeddings, BertSelfAttention, prune_linear_layer
from transformers.modeling_utils import PreTrainedModel

from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_albert import load_tf_weights_in_albert, AlbertEmbeddings, AlbertAttention, \
            AlbertLayer, AlbertLayerGroup, ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP, AlbertPreTrainedModel, \
            AlbertMLMHead, AlbertForMaskedLM


logger = logging.getLogger(__name__)


class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

        self.linear_size = 32
        self.num_items_to_keeps = [512, 288, 240, 224, 
                                   192, 176, 144, 96, 
                                   64, 64, 64, 64]  #0.18

        self.linear_1 = nn.ModuleList([nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size)
                                       ])

        self.linear_2 = nn.ModuleList([nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1)
                                      ])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, is_original=False, tokens_prob=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)
        if is_original:
            all_hidden_states = []
            for i in range(self.config.num_hidden_layers):
                # Number of layers in a hidden group
                layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

                # Index of the hidden group
                group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

                layer_group_output = self.albert_layer_groups[group_idx](
                    hidden_states,
                    attention_mask,
                    head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                )
                hidden_states = layer_group_output[0]
                all_hidden_states.append(hidden_states)
            
            outputs = (hidden_states, all_hidden_states)
            return outputs  

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)


        bsz = hidden_states.size(0)
        device = self.get_device_of(hidden_states)
        w = 0
        tot_zoom = None
        gate_loss = 0
        # TODO: 增加缩放操作

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
            )
            hidden_states = layer_group_output[0]

 
            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


ALBERT_START_DOCSTRING = r"""
"""

ALBERT_INPUTS_DOCSTRING = r"""
"""

@add_start_docstrings(
    "The bare ALBERT Model transformer outputting raw hidden-states without any specific head on top.",
    ALBERT_START_DOCSTRING,
)
class EnsembleAlbertModel(AlbertPreTrainedModel):

    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
            If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
            is a total of 4 different layers.

            These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
            while [2,3] correspond to the two inner groups of the second hidden layer.

            Any layer with in index other than [0,1,2,3] will result in an error.
            See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs



class AlbertPreTrainingHead(nn.Module):
    def __init__(self, config):
        super(AlbertPreTrainingHead, self).__init__()
        self.predictions = AlbertMLMHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class AlbertForPretraining(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForPretraining, self).__init__(config)

        self.albert = EnsembleAlbertModel(config)
        self.cls = AlbertPreTrainingHead(config)
        
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder, self.albert.embeddings.word_embeddings)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        sop_labels=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_outputs = outputs[0]
        pooled_output = outputs[1]

        prediction_scores, seq_relationship_score = self.cls(sequence_outputs, pooled_output)

        outputs = (prediction_scores, seq_relationship_score) + outputs[2:]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None and sop_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            prediction_scores = prediction_scores.view(-1, self.config.vocab_size)
            masked_lm_labels = masked_lm_labels.view(-1)

            masked_lm_loss = loss_fct(prediction_scores, masked_lm_labels)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), sop_labels.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

            outputs = (total_loss,) + outputs

        return outputs


@add_start_docstrings(
    """Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    ALBERT_START_DOCSTRING,
)
class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = EnsembleAlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ALBERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        """

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
