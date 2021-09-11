# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model. """


import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .activations import gelu, gelu_new, swish
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, prune_linear_layer
from .modeling_bert import BertEmbeddings, load_tf_weights_in_bert, BertLayerNorm, BertPreTrainedModel
from .modeling_bert import BertEncoder as OriBertEncoder
from .modeling_bert import BertModel as OriBertModel




logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",   
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
}


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}




class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        ori_attention_probs = attention_probs
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, ori_attention_probs)
        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        # if encoder_hidden_states is None:
        #     outputs = (context_layer, ori_attention_probs)
        # else:
        #     outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs
        return outputs



class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.linear_size = 32
        # self.num_items_to_keeps_512 = [512, 288, 240, 224,  192, 176, 144, 96, 64, 64, 64, 64]  #0.18
        self.num_items_to_keeps_512 =   [512, 192, 192, 160,  160, 160,  96, 96, 48, 48, 48, 48]

        # self.num_items_to_keeps_128 = [128, 64, 64, 64, 
        #                            64, 64, 48, 48, 
        #                            48, 48, 48, 48]  
        # self.num_items_to_keeps_128 = [128, 72, 60, 56, 48, 44, 36, 24, 16, 16, 16, 16]
        # self.num_items_to_keeps_128 = [128, 64, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32]   
        self.num_items_to_keeps_128 = [128, 64, 64, 32, 32, 16, 16, 16, 16, 16, 16, 16]   


        self.linear_1 = nn.ModuleList([nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                       nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size),
                                    #    nn.Linear(config.hidden_size, self.linear_size)
                                       ])

        self.linear_2 = nn.ModuleList([nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                       nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1),
                                    #    nn.Linear(self.linear_size, 1)
                                      ])


    def get_device_of(self, tensor):
        """
        Returns the device of the tensor.
        """
        if not tensor.is_cuda:
            return -1
        else:
            return tensor.get_device()

    def get_range_vector(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    def get_ones(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.FloatTensor(size, device=device).fill_(1)
        else:
            return torch.ones(size)

    def get_zeros(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).zero_()
        else:
            return torch.zeros(size, dtype=torch.long)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        seq_mask=None,
        is_original=False,
        tokens_prob=None
    ):

        if is_original:
            all_hidden_states = []

            for i, layer_module in enumerate(self.layer):
              
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
                )
                hidden_states = layer_outputs[0]
                all_hidden_states.append(hidden_states)


            outputs = (hidden_states, all_hidden_states)
            return outputs  
        num_items = hidden_states.size(1)
        if num_items==512:
            num_items_to_keeps = self.num_items_to_keeps_512
        else:
            num_items_to_keeps = self.num_items_to_keeps_128


        all_hidden_states = ()
        all_attentions = ()

        bsz = hidden_states.size(0)
        device = self.get_device_of(hidden_states)
        w = 0
        tot_zoom = None
        gate_loss = 0
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            num_items = hidden_states.size(1)
            num_items_to_keep = num_items_to_keeps[i]
            if num_items_to_keep < num_items:
                
                h1 = gelu(self.linear_1[w](hidden_states))
                h2 = (self.linear_2[w](h1)).squeeze(-1)
                gate = nn.LogSoftmax(dim=-1)(h2 -10000.0 * (1 - seq_mask))

                if tokens_prob is not None:
                    with torch.no_grad():
                        if tot_zoom is None:
                            t_p = tokens_prob[:,i-1,:]   # (bsz, 8, seq_len)
                        else:
                            t_p = torch.matmul(tot_zoom, tokens_prob[:, i-1, :].unsqueeze(-1)).squeeze(-1)
                    t_p = t_p / (torch.sum(t_p, dim=-1, keepdim=True)+1e-6)  #new add
    
                    tmp = gate * t_p
                    gate_loss += -torch.sum(tmp)
                                 
                with torch.no_grad():
                    gate_c = gate.clone()
                    gate_c[:, 0] += 1
                    top_values, top_indices = gate_c.topk(num_items_to_keep, 1)                    
                    
                    if device > -1:
                        zoomMatrix = torch.cuda.FloatTensor(bsz*num_items_to_keep, num_items, device=device).zero_()
                    else:
                        zoomMatrix = torch.zeros(bsz*num_items_to_keep, num_items)

                    idx = self.get_range_vector(bsz*num_items_to_keep, device)
                    zoomMatrix[idx, top_indices.view(-1)] = 1.
                    zoomMatrix = zoomMatrix.view(bsz, num_items_to_keep, num_items)

                    if tokens_prob is not None:
                        if tot_zoom is None:
                            tot_zoom = zoomMatrix
                        else:
                            tot_zoom = torch.matmul(zoomMatrix, tot_zoom)

                hidden_states = torch.matmul(zoomMatrix, hidden_states)
        
                seq_mask = torch.matmul(zoomMatrix, seq_mask.unsqueeze(-1))
                seq_mask = seq_mask.squeeze(-1)
                attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0

                w += 1


            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]
            attention_prob = layer_outputs[1]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if w>0:
            gate_loss =  gate_loss / hidden_states.size(0) / w  # found a bug here
        outputs = (hidden_states, gate_loss)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)



last_layer_val_grads =  None
# def save_grad(name):
#     def hook(grad):
#         last_layer_val_grads = grad
#     return hook
# def tensor_hook(grad):
#     global last_layer_val_grads
#     last_layer_val_grads = grad

class BertEncoderPred(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # self.layer_weight = nn.Parameter(torch.FloatTensor([[0.18866202235221863, 0.3252521753311157, 0.5090329051017761, 0.27924656867980957, 0.131510391831398, 0.16360431909561157, 0.18791458010673523, 0.4954243302345276, 0.14877276122570038, 0.21060949563980103, 0.23896948993206024, 0.2703886032104492], [0.390830397605896, 0.25819268822669983, 0.19549764692783356, 0.2955266237258911, 0.30468982458114624, 0.15731531381607056, 0.24801422655582428, 0.36835551261901855, 0.2445671111345291, 0.3653099834918976, 0.1789102554321289, 0.33908504247665405], [0.25420188903808594, 0.30061760544776917, 0.4025111198425293, 0.43941348791122437, 0.20631758868694305, 0.45903992652893066, 0.2928939461708069, 0.12526224553585052, 0.21876215934753418, 0.14259648323059082, 0.2025836855173111, 0.16099117696285248], [0.09600035101175308, 0.22791345417499542, 0.13377125561237335, 0.26869824528694153, 0.3182389438152313, 0.2199207991361618, 0.17356200516223907, 0.5647424459457397, 0.33468618988990784, 0.2213156372308731, 0.22064413130283356, 0.374656617641449], [0.6189013123512268, 0.07932592183351517, 0.32611551880836487, 0.19001562893390656, 0.19904546439647675, 0.19679220020771027, 0.15052561461925507, 0.29096361994743347, 0.22586533427238464, 0.2199377715587616, 0.20276588201522827, 0.37688907980918884], [0.35350558161735535, 0.30654582381248474, 0.33431798219680786, 0.23812203109264374, 0.28220972418785095, 0.2473078817129135, 0.39067909121513367, 0.25764554738998413, 0.2915409505367279, 0.1602514535188675, 0.1481601446866989, 0.34662652015686035], [0.5152267813682556, 0.20325428247451782, 0.07830598205327988, 0.2266678363084793, 0.3193471133708954, 0.3528963327407837, 0.31260907649993896, 0.23690740764141083, 0.15709900856018066, 0.23946380615234375, 0.1069423109292984, 0.402357280254364], [0.173335462808609, 0.20251615345478058, 0.14640697836875916, 0.41343462467193604, 0.2393549680709839, 0.3993820548057556, 0.5039165616035461, 0.32255396246910095, 0.1911507546901703, 0.047017280012369156, 0.31461697816848755, 0.1551644504070282], [0.2415853589773178, 0.19461742043495178, 0.41116440296173096, 0.14214524626731873, 0.1351793110370636, 0.3057677149772644, 0.17077146470546722, 0.146335169672966, 0.2067384123802185, 0.10760725289583206, 0.15061014890670776, 0.689312219619751], [0.24625644087791443, 0.48033544421195984, 0.21383267641067505, 0.2670658826828003, 0.17697808146476746, 0.26691266894340515, 0.260692298412323, 0.22048048675060272, 0.2241518646478653, 0.41932493448257446, 0.35888659954071045, 0.13256783783435822], [0.31023427844047546, 0.43290072679519653, 0.22183944284915924, 0.04939667508006096, 0.08180874586105347, 0.08247528970241547, 0.06069653853774071, 0.4379429519176483, 0.12447972595691681, 0.2690127193927765, 0.5530601739883423, 0.24897260963916779], [0.274791419506073, 0.20986537635326385, 0.29593682289123535, 0.3235595226287842, 0.2741956412792206, 0.1572055220603943, 0.1562899798154831, 0.3762766718864441, 0.49406498670578003, 0.3677799105644226, 0.19499877095222473, 0.06987588107585907]]))
        # self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.linear_2 = nn.Linear(config.hidden_size, 1)

    def get_device_of(self, tensor):
        """
        Returns the device of the tensor.
        """
        if not tensor.is_cuda:
            return -1
        else:
            return tensor.get_device()

    def get_range_vector(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    def get_ones(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.FloatTensor(size, device=device).fill_(1)
        else:
            return torch.ones(size)

    def get_zeros(self, size, device):
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).zero_()
        else:
            return torch.zeros(size, dtype=torch.long)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        seq_mask=None,
        is_original=False,
        tokens_prob=None
    ):
        start_index = 5
        if is_original:
            all_hidden_states = []

            for i, layer_module in enumerate(self.layer):
              
                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask
                )
                hidden_states = layer_outputs[0]
                all_hidden_states.append(hidden_states)


            outputs = (hidden_states, all_hidden_states)

            return outputs  

        all_hidden_states = ()
        all_attentions = ()
        # scores = []
        for i, layer_module in enumerate(self.layer[:start_index+1]):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, 
            )
            hidden_states = layer_outputs[0]
            attention_prob = layer_outputs[1]

            # attention_score = layer_outputs[1].detach()
            # attention_score = torch.sum(attention_score, dim=2)  #  bsz, head_num, token_num
            # weight = self.layer_weight[i].unsqueeze(0).unsqueeze(-1)
            # score = torch.sum(attention_score * weight, dim=1)  # bsz, head_num, token_num
            
            # scores.append(score)

        start_states = hidden_states

        # with torch.no_grad():
        #     for i, layer_module in enumerate(self.layer[start_index+1:], start_index+1):
        #         if self.output_hidden_states:
        #             all_hidden_states = all_hidden_states + (hidden_states,)

        #         layer_outputs = layer_module(
        #             hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, buqi_gate=None
        #         )
        #         hidden_states = layer_outputs[0]
        #         attention_score = layer_outputs[1].detach()

        #         attention_score = torch.sum(attention_score, dim=2)  #  bsz, head_num, token_num
        #         weight = self.layer_weight[i].unsqueeze(0).unsqueeze(-1)
        #         score = torch.sum(attention_score * weight, dim=1)  # bsz, token_num

        #         scores.append(score)

        # standared_score = sum(scores[3:])
        # standared_score = standared_score / torch.sum(standared_score, dim=-1, keepdim=True)


        # h1 = gelu(self.linear_1(start_states))
        # h2 = self.linear_2(h1).squeeze(-1)
        # gate = nn.LogSoftmax(dim=-1)(h2 -10000.0 * (1 - seq_mask))

        # if tokens_prob is not None:
        #     tmp = gate * seq_mask * tokens_prob #standared_score#tokens_prob
        #     gate_loss = -torch.sum(tmp) / gate.size(0)
        #     score = gate #tokens_prob
        # else:
        #     gate_loss = None
        #     score = gate

        # gate = tokens_prob
        gate = torch.sum(torch.sum(attention_prob, dim=2), dim=1)

        gate_loss = None

        bsz = hidden_states.size(0)
        num_items = hidden_states.size(1)
        num_items_to_keep = 32
        device = self.get_device_of(hidden_states)

        # remain_part                
        with torch.no_grad():
            gate_c = gate.clone()
            gate_c[:, 0] += 1
            top_values, top_indices = gate_c.topk(num_items_to_keep, 1)                    

            if device > -1:
                zoomMatrix = torch.cuda.FloatTensor(bsz*num_items_to_keep, num_items, device=device).zero_()
            else:
                zoomMatrix = torch.zeros(bsz*num_items_to_keep, num_items)
            idx = self.get_range_vector(bsz*num_items_to_keep, device)
            zoomMatrix[idx, top_indices.view(-1)] = 1.
            zoomMatrix = zoomMatrix.view(bsz, num_items_to_keep, num_items)#.transpose(1, 2)

        hidden_states = torch.matmul(zoomMatrix,  start_states)#self.batched_index_select(hidden_states, top_indices, flat_top_indices)

        seq_mask = torch.matmul(zoomMatrix, seq_mask.unsqueeze(-1))#self.batched_index_select(seq_mask.unsqueeze(-1), top_indices, flat_top_indices)
        seq_mask = seq_mask.squeeze(-1)
        attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0

        for i, layer_module in enumerate(self.layer[start_index+1:], start_index+1):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)


            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, 
            )
            hidden_states = layer_outputs[0]


            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, gate_loss)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)



class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score



BERT_START_DOCSTRING = r"""
"""

BERT_INPUTS_DOCSTRING = r"""
"""


class EnsembleBertModel(BertPreTrainedModel):
 
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder =  BertEncoder(config)#BertEncoderPred(config)#BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        is_original=False,
        score=None,
        tokens_prob=None
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
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
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        seq_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            seq_mask=seq_mask,
            is_original=is_original,
            tokens_prob=tokens_prob
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        next_sentence_label=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class EnsembleBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = EnsembleBertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING,
)
class EnsembleBertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = EnsembleBertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)



@add_start_docstrings(
    """Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    BERT_START_DOCSTRING,
)
class EnsembleBertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = EnsembleBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        is_original=False,
        tokens_prob=None
    ):

        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        if tokens_prob is not None:
            sizes = list(tokens_prob.size())
            tokens_prob = tokens_prob.view( [-1] + sizes[2:] )  

            
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            is_original=is_original,
            tokens_prob=tokens_prob
        )

        pooled_output = outputs[1]
        if tokens_prob is not None:
            gate_loss = outputs[2]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        if is_original:
            outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here
        else:
            outputs = (reshaped_logits,) + outputs[3:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

            if tokens_prob is not None:
                loss = loss +  gate_loss 

            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)



@add_start_docstrings(
    """Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING,
)
class EnsembleBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = EnsembleBertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForQuestionAnswering
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        assert answer == "a nice puppet"

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING,
)
class EnsembleBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = EnsembleBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        is_original=False,
        tokens_prob=None,
    ):
    
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            is_original=is_original,
            tokens_prob=tokens_prob,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if tokens_prob is not None:
            selector_loss = outputs[2]

        if is_original:
            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        else:
            outputs = (logits,) + outputs[3:]  # add hidden states and attention if they are here



        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (selector_loss,) + outputs

            # if tokens_prob is not None:
            #     loss = loss +  selector_loss 

            # outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

# Encoder part
# if i in [3]:
#     bsz = hidden_states.size(0)
#     num_items = hidden_states.size(1)
#     num_items_to_keep = 128
#     device = self.get_device_of(hidden_states)

#     attention_score = layer_outputs[1]
#     with torch.no_grad():
#         attention_score = torch.sum(attention_score, dim=1)  # head level
#         score = torch.sum(attention_score, dim=1)  # seq level

#     with torch.no_grad():
#         _, indices = torch.sort(score, descending=True)
#         top_indices = indices[:,:num_items_to_keep].contiguous().squeeze(-1)
#         top_indices, _ = torch.sort(top_indices)

#     # remain_part                
#     with torch.no_grad():
#         if device > -1:
#             zoomMatrix = torch.cuda.FloatTensor(bsz*num_items_to_keep, num_items, device=device).zero_()
#         else:
#             zoomMatrix = torch.zeros(bsz*num_items_to_keep, num_items)
#         idx = self.get_range_vector(bsz*num_items_to_keep, device)
#         zoomMatrix[idx, top_indices.view(-1)] = 1.
#         zoomMatrix = zoomMatrix.view(bsz, num_items_to_keep, num_items)#.transpose(1, 2)

#     hidden_states = torch.matmul(zoomMatrix, hidden_states)#self.batched_index_select(hidden_states, top_indices, flat_top_indices)

#     seq_mask = torch.matmul(zoomMatrix, seq_mask.unsqueeze(-1))#self.batched_index_select(seq_mask.unsqueeze(-1), top_indices, flat_top_indices)
#     seq_mask = seq_mask.squeeze(-1)
#     attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
#     attention_mask = (1.0 - attention_mask) * -10000.0



# class BertDecEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.output_attentions = config.output_attentions
#         self.output_hidden_states = config.output_hidden_states
#         self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
#         self.skip_layers = nn.ModuleList([BertLayer(config)])
#         #, BertLayer(config), BertLayer(config)])
#         self.gate_output_layers = nn.ModuleList([nn.Linear(config.hidden_size, 1)])#, 
#         # nn.Linear(config.hidden_size, 1), nn.Linear(config.hidden_size, 1)])

#     def get_device_of(self, tensor):
#         """
#         Returns the device of the tensor.
#         """
#         if not tensor.is_cuda:
#             return -1
#         else:
#             return tensor.get_device()

#     def get_range_vector(self, size, device):
#         """
#         Returns a range vector with the desired size, starting at 0. The CUDA implementation
#         is meant to avoid copy data from CPU to GPU.
#         """
#         if device > -1:
#             return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
#         else:
#             return torch.arange(0, size, dtype=torch.long)

#     def get_ones(self, size, device):
#         """
#         Returns a range vector with the desired size, starting at 0. The CUDA implementation
#         is meant to avoid copy data from CPU to GPU.
#         """
#         if device > -1:
#             return torch.cuda.FloatTensor(size, device=device).fill_(1)
#         else:
#             return torch.ones(size)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         seq_mask=None
#     ):
#         all_hidden_states = ()
#         all_attentions = ()
#         # final_state = torch.zeros_like(hidden_states)
#         buqi_gate = None
#         ori_attention_mask = attention_mask
#         ori_seq_mask = seq_mask
#         for i, layer_module in enumerate(self.layer):
#             if self.output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             if i in [12]:#[4, 8, 11]:
#                 hidden_states = (hidden_states  - ori_hidden_states) * buqi_gate + ori_hidden_states
#                 hidden_states = torch.matmul(zoomMatrix_T, hidden_states)
#                 hidden_states += middle_state
#                 attention_mask = ori_attention_mask
#                 seq_mask = ori_seq_mask

#             layer_outputs = layer_module(
#                 hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, buqi_gate=None
#             )
#             hidden_states = layer_outputs[0]

#             if i in [3]:#[0, 4, 8]:
#                 layer_idx = i//4
#                 bsz = hidden_states.size(0)
#                 num_items = hidden_states.size(1)
#                 num_items_to_keep = 256#128
#                 num_items_to_skip = num_items - num_items_to_keep
#                 device = self.get_device_of(hidden_states)

#                 # attention_score = layer_outputs[1]
#                 # with torch.no_grad():
#                 #     attention_score = torch.sum(attention_score, dim=1)  # head level
#                 #     score = torch.sum(attention_score, dim=1)  # seq level
#                 gate = self.gate_output_layers[layer_idx](hidden_states).squeeze(-1)
#                 gate = gate - 10000.0 * (1-seq_mask)
#                 score  = nn.Softmax(dim=-1)(gate)


#                 with torch.no_grad():
#                     _, indices = torch.sort(score, descending=True)
#                     skip_indices = indices[:, num_items_to_keep:].contiguous().squeeze(-1)
#                     top_indices = indices[:,:num_items_to_keep].contiguous().squeeze(-1)

#                 with torch.no_grad():
#                     if device > -1:
#                         skipZoomMatrix = torch.cuda.FloatTensor(bsz*num_items_to_skip, num_items, device=device).zero_()
#                     else:
#                         skipZoomMatrix = torch.zeros(bsz*num_items_to_skip, num_items)
                        
#                     idx = self.get_range_vector(bsz*num_items_to_skip, device)
#                     skipZoomMatrix[idx, skip_indices.view(-1)] = 1.
#                     skipZoomMatrix = skipZoomMatrix.view(bsz, num_items_to_skip, num_items)#.transpose(1, 2)
#                     skipZoomMatrix_T = skipZoomMatrix.transpose(1, 2)

#                 skip_hidden_states = torch.matmul(skipZoomMatrix, hidden_states)
                
#                 # (batch_size, hidden_size, 1)
#                 skip_seq_mask = torch.matmul(skipZoomMatrix, seq_mask.unsqueeze(-1))

#                 encoder_attention_mask = torch.matmul(skip_seq_mask, seq_mask.unsqueeze(1))
#                 encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
#                 encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0

#                 # (batch_size, num_items_to_skip, hidden_size)
#                 skip_output = self.skip_layers[layer_idx](skip_hidden_states, encoder_hidden_states=hidden_states, encoder_attention_mask=encoder_attention_mask)
#                 skip_hidden_states = skip_output[0]
#                 middle_state = torch.matmul(skipZoomMatrix_T, skip_hidden_states)

#                 # remain_part                
#                 with torch.no_grad():
#                     if device > -1:
#                         zoomMatrix = torch.cuda.FloatTensor(bsz*num_items_to_keep, num_items, device=device).zero_()
#                     else:
#                         zoomMatrix = torch.zeros(bsz*num_items_to_keep, num_items)
#                     idx = self.get_range_vector(bsz*num_items_to_keep, device)
#                     zoomMatrix[idx, top_indices.view(-1)] = 1.
#                     zoomMatrix = zoomMatrix.view(bsz, num_items_to_keep, num_items)#.transpose(1, 2)
#                     zoomMatrix_T = zoomMatrix.transpose(1,2)

#                 hidden_states = torch.matmul(zoomMatrix, hidden_states)#self.batched_index_select(hidden_states, top_indices, flat_top_indices)
#                 ori_hidden_states = hidden_states

#                 seq_mask = torch.matmul(zoomMatrix, seq_mask.unsqueeze(-1))#self.batched_index_select(seq_mask.unsqueeze(-1), top_indices, flat_top_indices)

#                 gate = torch.matmul(zoomMatrix, gate.unsqueeze(-1))#self.batched_index_select(seq_mask.unsqueeze(-1), top_indices, flat_top_indices)
#                 buqi_gate = (1 - gate.detach() + gate) * seq_mask

#                 seq_mask = seq_mask.squeeze(-1)
#                 attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
#                 attention_mask = (1.0 - attention_mask) * -10000.0


#             if self.output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         # Add last layer
#         if self.output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         outputs = (hidden_states, )
#         if self.output_hidden_states:
#             outputs = outputs + (all_hidden_states,)
#         if self.output_attentions:
#             outputs = outputs + (all_attentions,)
#         return outputs  # last-layer hidden state, (all hidden states), (all attentions)


# class BertEncoderSandGlass(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.output_attentions = config.output_attentions
#         self.output_hidden_states = config.output_hidden_states
#         self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
#         self.skip_layers = nn.ModuleList([BertLayer(config), BertLayer(config), 
#                                 BertLayer(config), BertLayer(config)])
#         self.gate_output_layers = nn.ModuleList([nn.Linear(config.hidden_size, 1), 
#                                     nn.Linear(config.hidden_size, 1), 
#                                     nn.Linear(config.hidden_size, 1), 
#                                     nn.Linear(config.hidden_size, 1)])

#         self.layer_size = [512, 416, 320, 224, 128]
#     def get_device_of(self, tensor):
#         """
#         Returns the device of the tensor.
#         """
#         if not tensor.is_cuda:
#             return -1
#         else:
#             return tensor.get_device()

#     def get_range_vector(self, size, device):
#         """
#         Returns a range vector with the desired size, starting at 0. The CUDA implementation
#         is meant to avoid copy data from CPU to GPU.
#         """
#         if device > -1:
#             return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
#         else:
#             return torch.arange(0, size, dtype=torch.long)

#     def get_ones(self, size, device):
#         """
#         Returns a range vector with the desired size, starting at 0. The CUDA implementation
#         is meant to avoid copy data from CPU to GPU.
#         """
#         if device > -1:
#             return torch.cuda.FloatTensor(size, device=device).fill_(1)
#         else:
#             return torch.ones(size)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         seq_mask=None,
#     ):
#         all_hidden_states = ()
#         all_attentions = ()

#         buqi_gate = None
#         # ori_attention_mask = attention_mask
#         # ori_seq_mask = seq_mask

#         middle_states = []
#         zoomMatrixs = []
#         attention_masks = []
#         ori_hidden_states = []
#         buqi_gates = []
#         for i, layer_module in enumerate(self.layer):
#             if self.output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             if i in [8, 9, 10, 11]:
#                 ori_hidden_state = ori_hidden_states.pop()
#                 buqi_gate = buqi_gates.pop()
#                 hidden_states = (hidden_states  - ori_hidden_state) * buqi_gate + ori_hidden_state
#                 zooM = zoomMatrixs.pop()          
#                 hidden_states = torch.matmul(zooM, hidden_states)
#                 hidden_states += middle_states.pop()
#                 attention_mask = attention_masks.pop()


#             layer_outputs = layer_module(
#                 hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask, buqi_gate=None
#             )
#             hidden_states = layer_outputs[0]

#             if i in [0, 1, 2, 3]:
#                 attention_masks.append(attention_mask)
#                 layer_idx = i
#                 bsz = hidden_states.size(0)
#                 num_items = hidden_states.size(1)
#                 num_items_to_keep = self.layer_size[i+1]
#                 num_items_to_skip = num_items - num_items_to_keep
#                 device = self.get_device_of(hidden_states)
#                 # attention_score = layer_outputs[1]
#                 # with torch.no_grad():
#                 #     attention_score = torch.sum(attention_score, dim=1)  # head level
#                 #     score = torch.sum(attention_score, dim=1)  # seq level
#                 gate_layer = self.gate_output_layers[layer_idx]
#                 gate = gate_layer(hidden_states).squeeze(-1)
#                 gate = gate - 10000.0 * (1-seq_mask)
#                 score  = nn.Softmax(dim=-1)(gate)


#                 with torch.no_grad():
#                     _, indices = torch.sort(score, descending=True)
#                     skip_indices = indices[:, num_items_to_keep:].contiguous().squeeze(-1)
#                     top_indices = indices[:,:num_items_to_keep].contiguous().squeeze(-1)

#                 with torch.no_grad():
#                     if device > -1:
#                         skipZoomMatrix = torch.cuda.FloatTensor(bsz*num_items_to_skip, num_items, device=device).zero_()
#                     else:
#                         skipZoomMatrix = torch.zeros(bsz*num_items_to_skip, num_items)
                        
#                     idx = self.get_range_vector(bsz*num_items_to_skip, device)
#                     skipZoomMatrix[idx, skip_indices.view(-1)] = 1.
#                     skipZoomMatrix = skipZoomMatrix.view(bsz, num_items_to_skip, num_items)#.transpose(1, 2)
#                     skipZoomMatrix_T = skipZoomMatrix.transpose(1, 2)

#                 skip_hidden_states = torch.matmul(skipZoomMatrix, hidden_states)
                
#                 # (batch_size, hidden_size, 1)
#                 skip_seq_mask = torch.matmul(skipZoomMatrix, seq_mask.unsqueeze(-1))

#                 encoder_attention_mask = torch.matmul(skip_seq_mask, seq_mask.unsqueeze(1))
#                 encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
#                 encoder_attention_mask = (1.0 - encoder_attention_mask) * -10000.0

#                 # (batch_size, num_items_to_skip, hidden_size)
#                 skip_output = self.skip_layers[layer_idx](skip_hidden_states, encoder_hidden_states=hidden_states, encoder_attention_mask=encoder_attention_mask)
#                 skip_hidden_states = skip_output[0]
#                 middle_state = torch.matmul(skipZoomMatrix_T, skip_hidden_states)
#                 middle_states.append(middle_state)

#                 # remain_part                
#                 with torch.no_grad():
#                     if device > -1:
#                         zoomMatrix = torch.cuda.FloatTensor(bsz*num_items_to_keep, num_items, device=device).zero_()
#                     else:
#                         zoomMatrix = torch.zeros(bsz*num_items_to_keep, num_items)

#                     idx = self.get_range_vector(bsz*num_items_to_keep, device)
#                     zoomMatrix[idx, top_indices.view(-1)] = 1.
#                     zoomMatrix = zoomMatrix.view(bsz, num_items_to_keep, num_items)#.transpose(1, 2)
#                     zoomMatrix_T = zoomMatrix.transpose(1, 2)
#                 zoomMatrixs.append(zoomMatrix_T)
#                 hidden_states = torch.matmul(zoomMatrix, hidden_states)#self.batched_index_select(hidden_states, top_indices, flat_top_indices)
#                 ori_hidden_state = hidden_states
#                 ori_hidden_states.append(ori_hidden_state)

#                 seq_mask = torch.matmul(zoomMatrix, seq_mask.unsqueeze(-1))#self.batched_index_select(seq_mask.unsqueeze(-1), top_indices, flat_top_indices)

#                 gate = torch.matmul(zoomMatrix, gate.unsqueeze(-1))#self.batched_index_select(seq_mask.unsqueeze(-1), top_indices, flat_top_indices)
#                 buqi_gate = (1 - gate.detach() + gate) * seq_mask
#                 buqi_gates.append(buqi_gate)

#                 seq_mask = seq_mask.squeeze(-1)
#                 attention_mask = seq_mask.unsqueeze(1).unsqueeze(2)
#                 attention_mask = (1.0 - attention_mask) * -10000.0


#             if self.output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         # Add last layer
#         if self.output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         outputs = (hidden_states, )
#         if self.output_hidden_states:
#             outputs = outputs + (all_hidden_states,)
#         if self.output_attentions:
#             outputs = outputs + (all_attentions,)
#         return outputs  # last-layer hidden state, (all hidden states), (all attentions)
