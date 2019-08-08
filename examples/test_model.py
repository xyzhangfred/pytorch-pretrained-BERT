###load model, fix the weights, test the representation on pos_tag classification.
#3h work!

import argparse
import os
import random
import logging, datetime
from tqdm import tqdm, trange
from io import open
from scipy.misc import logsumexp

import flair

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (GPT2Config,GPT2Model, GPT2LMHeadModel,AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG, filename='adv_gpt2.log')
logger = logging.getLogger(__name__)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class TaggedCorpus(object):
    def __init__(self, path):

        self.dictionary = Dictionary()
        self.pos_dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'tagged_train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'tagged_valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'tagged_test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file into """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                if len(line.strip().split()[::2]) > 3:
                    words = ['<sos>'] + line.strip().split()[::2] + ['<eos>']
                    pos_tags = ['<SOS>'] + line.strip().split()[1::2] + ['<EOS>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
                    for tag in pos_tags:
                        self.pos_dictionary.add_word(tag)
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            pos_ids = torch.LongTensor(tokens)
            token = 0
            pos_token = 0
            for line in f:
                if len(line.strip().split()[::2]) > 3:
                #print (line.strip().split())
                    words = ['<sos>']+line.strip().split()[::2] + ['<eos>']
                    pos_tags = ['<SOS>'] + line.strip().split()[1::2] + ['<EOS>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1
                    for i,tag in enumerate(pos_tags):
                        pos_ids[pos_token] =self.pos_dictionary.word2idx[tag]
                        pos_token += 1
        return ids,pos_ids


def load_tokenize_and_batchify(data_dir = '../SemSynLSTM/word_language_model/data/wikitext-2/', input_len = 128):
    """
    load dataset and return train, val, test dataset

    """
    tensor_datasets = []

    corpus = TaggedCorpus(data_dir)
    train_data = corpus.train
    val_data = corpus.valid
    test_data = corpus.test

    for dataset in [train_data, val_data, test_data]:
        ##divide data by batch, truncate to fit into batches
        n_batch = len(dataset[0]) // input_len
        input_ids = dataset[0][: n_batch * input_len].reshape(n_batch, input_len)
        pos_ids = dataset[1][: n_batch * input_len].reshape(n_batch, input_len)
        all_inputs = (input_ids, pos_ids)
        tensor_datasets.append(tuple(t for t in all_inputs))

    return tensor_datasets[0], tensor_datasets[1],tensor_datasets[2], corpus.dictionary, corpus.pos_dictionary