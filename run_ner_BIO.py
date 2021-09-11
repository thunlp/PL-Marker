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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
from collections import defaultdict
import re
import shutil

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  BertForTokenClassification,
                                  get_linear_schedule_with_warmup,
                                  AdamW, 
                                  RobertaForTokenClassification,
                                  AlbertConfig,
                                  AlbertTokenizer
                                  )

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
import json
import pickle
import numpy as np
import unicodedata

import itertools
import seqeval.metrics
import math
import timeit

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForTokenClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
}

class CoNLL03Dataset(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False):

        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                file_path = os.path.join(args.data_dir, args.test_file)
            else:
                file_path = os.path.join(args.data_dir, args.dev_file)

        assert os.path.isfile(file_path)

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_mention_length = args.max_mention_length
        self.max_pair_length = args.max_pair_length

    
        self.evaluate = evaluate
        self.doc_stride = args.doc_stride
        self.args = args
        self.model_type = args.model_type
        self.ori_label_list = ["MISC", "PER", "ORG", "LOC"]
        self.label_list = ['O']
        for label in self.ori_label_list:
            self.label_list.append('B-' + label)
            self.label_list.append('I-' + label)
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        self.max_entity_length = args.max_pair_length * 2

        self.data_info = self._read_data(file_path)

        self.initialize()

    def _read_data(self, input_file):
        data = []
        words = []
        labels = []
        sentence_boundaries = []
        with open(input_file) as f:
            for line in tqdm(f):
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    if words:

                        data.append((words, labels, sentence_boundaries))
                        if self.args.output_dir.find('test')!=-1:
                            if len(data) == 5:
                                return data

                        assert sentence_boundaries[0] == 0
                        assert sentence_boundaries[-1] == len(words)
                        words = []
                        labels = []
                        sentence_boundaries = []
                    continue

                if not line:
                    if not sentence_boundaries or len(words) != sentence_boundaries[-1]:
                        sentence_boundaries.append(len(words))
                else:
                    parts = line.split(" ")
                    words.append(parts[0])
                    labels.append(parts[-1])

        if words:
            data.append((words, labels, sentence_boundaries))
        return data

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def initialize(self):
        tokenizer = self.tokenizer
        self.data = []

        max_num_subwords = self.max_seq_length - 2

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)
        maxR = 0
        for example_index, (words, labels, sentence_boundaries) in enumerate(self.data_info):
            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]

            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]
            assert(len(labels) == len(words))

            for n in range(len(subword_sentence_boundaries) - 1):
                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)

                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
                target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
                
                word_sent_start, word_sent_end = sentence_boundaries[n: n+2]
                sentence_labels = labels[word_sent_start : word_sent_end]

                start_positions = []
                label_ids = [] 
                for entity_start in range(left_context_length, left_context_length + sentence_length):
                    doc_entity_start = entity_start + doc_offset

                    if doc_entity_start in subword_start_positions:
                        token_idx = subword2token[doc_entity_start]
                        label_ids.append(self.label_map[labels[token_idx]])
                        start_positions.append(entity_start+1)
                    else:
                        label_ids.append(-1)
                label_ids = [-1] * (left_context_length+1) + label_ids + [-1] * (right_context_length+1)
                assert(len(sentence_labels) == len(start_positions))
                assert(len(label_ids) == len(target_tokens))

                item = {
                    'sentence': target_tokens,
                    'sentence_labels': sentence_labels,
                    'label_ids': label_ids,
                    'start_positions': start_positions,
                    # 'example_index': (example_index, n)
                }                

                self.data.append(item)

        logger.info('maxR: %s', maxR)

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # left_ctx = entry['left_ctx']
        sentence_labels = entry['sentence_labels']
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)

        input_ids += [0] * (self.max_seq_length - len(input_ids))

        attention_mask = [1] * L + [0] * (self.max_seq_length - L)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)


        labels = entry['label_ids'] + [-1] * (self.max_seq_length - L)


        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(labels, dtype=torch.int64),
        ]       

        if self.evaluate:
            item.append(sentence_labels)
            item.append(entry['start_positions'])
            # item.append(left_ctx)

        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 2
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields
 



class OntonotesDataset(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False):
        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                file_path = os.path.join(args.data_dir, args.test_file)
            else:
                file_path = os.path.join(args.data_dir, args.dev_file)

        assert os.path.isfile(file_path)

        self.file_path = file_path
                
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_mention_length = args.max_mention_length

        self.evaluate = evaluate
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type

        self.ori_label_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

        self.label_list = ['O']
        for label in self.ori_label_list:
            self.label_list.append('B-' + label)
            self.label_list.append('I-' + label)
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        self.max_pair_length = args.max_pair_length

        self.max_entity_length = args.max_pair_length * 2
        self.initialize()

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False 

    def get_original_token(self, token):
        escape_to_original = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        if token in escape_to_original:
            token = escape_to_original[token]
        return token
        
    
    def initialize(self):
        tokenizer = self.tokenizer
        max_num_subwords = self.max_seq_length - 2

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        self.data = []

        maxL = 0

        for l_idx, line in enumerate(f):
            data = json.loads(line)
            if self.args.output_dir.find('test')!=-1:
                if len(self.data) > 5:
                    break

            sentences = data['sentences']
            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    sentences[i][j] = self.get_original_token(sentences[i][j])
            
            ners = data['ner']

            sentence_boundaries = [0]
            words = []
            L = 0
            for i in range(len(sentences)):
                L += len(sentences[i])
                sentence_boundaries.append(L)
                words += sentences[i]

            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]
            maxL = max(len(tokens), maxL)
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]

            for n in range(len(subword_sentence_boundaries) - 1):
                sentence_ners = ners[n]

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]
                word_sent_start, word_sent_end = sentence_boundaries[n: n+2]

                sentence_labels = ["O"] * (word_sent_end-word_sent_start)
                for start, end, label in sentence_ners:
                    start -= word_sent_start
                    end -= word_sent_start

                    sentence_labels[start] = "B-" + label
                    if end - start > 0:
                        sentence_labels[start + 1 : end+1] = ["I-" + label] * (end - start)

                # convert IOB2 -> IOB1
                prev_type = None
                for n, label in enumerate(sentence_labels):
                    if (label[0] == "B") and (label[2:] != prev_type):
                        sentence_labels[n] = "I" + label[1:]
                    prev_type = label[2:]


                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)
                if self.args.output_dir.find('ctx0')!=-1:
                    left_context_length = right_context_length = 0 # for debug

                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
                assert(len(target_tokens)<=max_num_subwords)
                target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
                

                start_positions = []
                label_ids = [] 
                for entity_start in range(left_context_length, left_context_length + sentence_length):
                    doc_entity_start = entity_start + doc_offset

                    if doc_entity_start in subword_start_positions:
                        token_idx = subword2token[doc_entity_start]
                        label_ids.append(self.label_map[sentence_labels[ token_idx - word_sent_start ]])
                        start_positions.append(entity_start+1)
                    else:
                        label_ids.append(-1)
                label_ids = [-1] * (left_context_length+1) + label_ids + [-1] * (right_context_length+1)
                assert(len(sentence_labels) == len(start_positions))
                assert(len(label_ids) == len(target_tokens))


                item = {
                    'sentence': target_tokens,
                    'sentence_labels': sentence_labels,
                    'label_ids': label_ids,
                    'start_positions': start_positions,
                }                

                self.data.append(item)



    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # left_ctx = entry['left_ctx']
        sentence_labels = entry['sentence_labels']
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)

        input_ids += [0] * (self.max_seq_length - len(input_ids))

        attention_mask = [1] * L + [0] * (self.max_seq_length - L)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)


        labels = entry['label_ids'] + [-1] * (self.max_seq_length - L)


        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(labels, dtype=torch.int64),
        ]       

        if self.evaluate:
            item.append(sentence_labels)
            item.append(entry['start_positions'])

        return item




class FewNERDataset(Dataset):
    def __init__(self, tokenizer, args=None, evaluate=False, do_test=False):

        if not evaluate:
            file_path = os.path.join(args.data_dir, args.train_file)
        else:
            if do_test:
                file_path = os.path.join(args.data_dir, args.test_file)
            else:
                file_path = os.path.join(args.data_dir, args.dev_file)

        assert os.path.isfile(file_path)

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_mention_length = args.max_mention_length
        self.max_pair_length = args.max_pair_length
        self.max_entity_length = args.max_pair_length * 2
        self.evaluate = evaluate
        self.file_path = file_path
        self.doc_stride = args.doc_stride
        self.args = args
        self.model_type = args.model_type
        ori_label_list = ['art-broadcastprogram', 'art-film', 'art-music', 'art-other', 'art-painting', 'art-writtenart', 'building-airport', 'building-hospital', 'building-hotel', 'building-library', 'building-other', 'building-restaurant', 'building-sportsfacility', 'building-theater', 'event-attack/battle/war/militaryconflict', 'event-disaster', 'event-election', 'event-other', 'event-protest', 'event-sportsevent', 'location-GPE', 'location-bodiesofwater', 'location-island', 'location-mountain', 'location-other', 'location-park', 'location-road/railway/highway/transit', 'organization-company', 'organization-education', 'organization-government/governmentagency', 'organization-media/newspaper', 'organization-other', 'organization-politicalparty', 'organization-religion', 'organization-showorganization', 'organization-sportsleague', 'organization-sportsteam', 'other-astronomything', 'other-award', 'other-biologything', 'other-chemicalthing', 'other-currency', 'other-disease', 'other-educationaldegree', 'other-god', 'other-language', 'other-law', 'other-livingthing', 'other-medical', 'person-actor', 'person-artist/author', 'person-athlete', 'person-director', 'person-other', 'person-politician', 'person-scholar', 'person-soldier', 'product-airplane', 'product-car', 'product-food', 'product-game', 'product-other', 'product-ship', 'product-software', 'product-train', 'product-weapon']
        self.ori_label_list = [x.replace('-', '_') for x in ori_label_list]

        self.label_list = ['O']
        for label in self.ori_label_list:
            self.label_list.append('I-' + label)
        self.label_map = {label: i for i, label in enumerate(self.label_list)}

        self.data_info = self._read_data(file_path)
        if self.args.output_dir.find('test')!=-1:
            self.data_info = self.data_info[:5]

        self.initialize()

    def _read_data(self, input_file):
        data = []
        words = []
        labels = []
        sentence_boundaries = []
        prev_type = None
        with open(input_file) as f:
            for line in tqdm(f):
                line = line.rstrip()
                if not line:

                    if words:
                        sentence_boundaries = [0, len(words)]
                        data.append((words, labels, sentence_boundaries))

                    words = []
                    labels = []
                    prev_type = None

                else:
                    parts = line.split("\t")
                    words.append(parts[0])
                    label = parts[-1]
                    if label!='O':
                        label = parts[-1].replace('-', '_')
                        label = 'I-' + label

                    prev_type = label[2:]
                    labels.append(label)

        if words:
            data.append((words, labels, sentence_boundaries))
        # data = data[:100]
        print (len(data))
        return data


    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def initialize(self):
        tokenizer = self.tokenizer
        self.data = []

        max_num_subwords = self.max_seq_length - 2
        label_map = {label: i for i, label in enumerate(self.label_list)}

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            subwords =  tokenizer.tokenize(text)
            if len(text)>0 and len(subwords)==0: # for \u2063
                subwords = ['[UNK]']
            return subwords
 
        for example_index, (words, labels, sentence_boundaries) in tqdm(enumerate(self.data_info), disable=self.args.local_rank not in [-1, 0],  desc="Processing"):

            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]

            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]


            for n in range(len(subword_sentence_boundaries) - 1):
                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)

                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]
                target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
                
                word_sent_start, word_sent_end = sentence_boundaries[n: n+2]
                sentence_labels = labels[word_sent_start : word_sent_end]

                start_positions = []
                label_ids = [] 
                for entity_start in range(left_context_length, left_context_length + sentence_length):
                    doc_entity_start = entity_start + doc_offset

                    if doc_entity_start in subword_start_positions:
                        token_idx = subword2token[doc_entity_start]
                        label_ids.append(self.label_map[labels[token_idx]])
                        start_positions.append(entity_start+1)
                    else:
                        label_ids.append(-1)
                label_ids = [-1] * (left_context_length+1) + label_ids + [-1] * (right_context_length+1)
                assert(len(sentence_labels) == len(start_positions))
                assert(len(label_ids) == len(target_tokens))

                item = {
                    'sentence': target_tokens,
                    'sentence_labels': sentence_labels,
                    'label_ids': label_ids,
                    'start_positions': start_positions,
                }                

                self.data.append(item)


    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence_labels = entry['sentence_labels']
        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])
        L = len(input_ids)

        input_ids += [0] * (self.max_seq_length - len(input_ids))

        attention_mask = [1] * L + [0] * (self.max_seq_length - L)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)

        labels = entry['label_ids'] + [-1] * (self.max_seq_length - L)

        item = [torch.tensor(input_ids),
                attention_mask,
                torch.tensor(labels, dtype=torch.int64),
        ]       

        if self.evaluate:
            item.append(sentence_labels)
            item.append(entry['start_positions'])

        return item



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)



def train(args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/"+args.data_dir[max(args.data_dir.rfind('/'),0):]+"_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.data_dir.find('fewnerd')!=-1:
        train_dataset = FewNERDataset(tokenizer=tokenizer, args=args)
    elif args.data_dir.find('ontonotes')!=-1:
        train_dataset = OntonotesDataset(tokenizer=tokenizer, args=args)
    else:
        train_dataset = CoNLL03Dataset(tokenizer=tokenizer, args=args)
                            
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2*int(args.output_dir.find('test')==-1))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)  ], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon) 
    if args.warmup_steps==-1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = 0

    for _ in train_iterator:
        # if args.randstart:
        #     if _>0:
        #         train_dataset.initialize()

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                    #   'position_ids':   batch[2],
                      'labels':         batch[2],
                      }
                      
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        f1 = results['f1']
                        tb_writer.add_scalar('f1', f1, global_step)

                        if f1 > best_f1:
                            best_f1 = f1
                            print ('Best F1', best_f1)
                        else:
                            update = False
                    if update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, tokenizer, prefix="", do_test=False):

    eval_output_dir = args.output_dir
    results = {}
    if args.data_dir.find('fewnerd')!=-1:
        eval_dataset = FewNERDataset(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test)
    elif args.data_dir.find('ontonotes')!=-1:
        eval_dataset = OntonotesDataset(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test)
    else:
        eval_dataset = CoNLL03Dataset(tokenizer=tokenizer, args=args, evaluate=True, do_test=do_test)


    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    label_list = eval_dataset.label_list

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=CoNLL03Dataset.collate_fn, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)


    all_predictions = defaultdict(dict)
    model.eval()

    final_labels = []
    final_predictions = []
    num_ori_labels = len(eval_dataset.ori_label_list)

    start_time = timeit.default_timer() 
    tot_candidates = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        original_labels = batch[-2]
        start_positions = batch[-1]

        batch = tuple(t.to(args.device) for t in batch[:-2])
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2]
                      }

            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            for i in range(len(start_positions)):
                sentence_labels = original_labels[i]
                start_position = start_positions[i]
                prob = probs[i]
                pred = preds[i]
                predicted_sequence = []

                all_probs = []
                for j in range(len(start_position)):
                    pred_label = label_list[pred[ start_position[j]  ]]
                    predicted_sequence.append(pred_label)
                    if args.data_dir.find('fewnerd')!=-1:
                        p = prob[start_position[j]][1:]
                    else:
                        p = prob[start_position[j]]
                        p = p[1:1+num_ori_labels] + p[1+num_ori_labels:]

                    all_probs.append(p)
                
                if args.output_candidates and do_test:
                    # if args.data_dir.find('fewnerd')!=-1:
                    span_prob = []
                    for l in range(len(all_probs)):
                        if l == 0:
                            prob = np.ones(num_ori_labels, dtype=np.float32)
                            # prob = torch.ones(num_ori_labels).to(all_probs[0])

                        else:
                            prob = 1 - all_probs[l-1]
                        for r in range(l, len(all_probs)):
                            if r-l+1 > args.max_mention_ori_length:
                                break
                            prob = prob * all_probs[r]

                            if r+1 != len(all_probs):
                                final_prob = prob * (1 - all_probs[r+1])
                            else:
                                final_prob = prob

                            span_prob.append( ((l, r), final_prob.max())  )                     
    
                    span_prob.sort(key=lambda x: -x[1])
                    candidates = [x[0] for x in span_prob]
                    candidates = candidates[:256]
                    tot_candidates.append(candidates)

                assert(len(predicted_sequence)==len(sentence_labels))
                # convert IOB2 -> IOB1
                prev_type = None
                for n, label in enumerate(predicted_sequence):
                    if (label[0] == "B") and (label[2:] != prev_type):
                        predicted_sequence[n] = "I" + label[1:]
                    prev_type = label[2:]

                final_predictions.append(predicted_sequence)
                final_labels.append(sentence_labels)
 
    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime,  len(eval_dataset) / evalTime)

    assert len(final_predictions) == len(final_labels)
    f1 = seqeval.metrics.f1_score(final_labels, final_predictions)
    precision_score = seqeval.metrics.precision_score(final_labels, final_predictions)
    recall_score = seqeval.metrics.recall_score(final_labels, final_predictions)

    results = {'f1': f1, 'precision': precision_score, 'recall_score': recall_score}
    logger.info("Result: %s", json.dumps(results))

    if args.output_candidates and do_test:
        output_w = open(os.path.join(args.output_dir, "cands.json"), 'w')
        json.dump(tot_candidates,  output_w)

    return results



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='data', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run test on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_b2", default=0.999, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file",  default="eng.train", type=str)
    parser.add_argument("--dev_file",  default="eng.testa", type=str)
    parser.add_argument("--test_file",  default="eng.testb", type=str)

    parser.add_argument('--alpha', type=float, default=1,  help="")

    parser.add_argument('--max_pair_length', type=int, default=256,  help="")
    parser.add_argument('--max_mention_length', type=int, default=12,  help="")
    parser.add_argument('--max_mention_ori_length', type=int, default=8,  help="")
    parser.add_argument('--doc_stride', type=int, default=0,  help="")
    parser.add_argument('--output_candidates', action='store_true')
    
    
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        if not os.path.exists(path):
            os.mkdir(path)

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    if args.do_train and args.local_rank in [-1, 0] and args.output_dir.find('test')==-1:
        create_exp_dir(args.output_dir, scripts_to_save=['run_ner.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_roberta.py'])


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.data_dir.find('fewnerd')!=-1:
        args.num_labels = 67
    elif args.data_dir.find('ontonotes')!=-1:
        args.num_labels = 18*2+1
    else:
        args.num_labels = 9

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.num_labels)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,  do_lower_case=args.do_lower_case)

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)


    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0

    if args.do_train and args.model_type.startswith('albert'):
        special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    # Training
    if args.do_train:
        global_step, tr_loss, best_f1 = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        update = True
        if args.evaluate_during_training:
            results = evaluate(args, model, tokenizer)
            f1 = results['f1']
            if f1 > best_f1:
                best_f1 = f1
                print ('Best F1', best_f1)
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]

        WEIGHTS_NAME = 'pytorch_model.bin'

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate on test set")

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""

            model = model_class.from_pretrained(checkpoint, config=config)

            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, do_test=True)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.local_rank in [-1, 0]:
        print (results)

        output_eval_file = os.path.join(args.output_dir, "results.json")
        json.dump(results, open(output_eval_file, "w"))


if __name__ == "__main__":
    main()
