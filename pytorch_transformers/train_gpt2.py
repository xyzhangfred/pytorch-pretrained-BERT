# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import random
import logging
import json
from tqdm import tqdm, trange
from io import open
from scipy.special import logsumexp
import wandb
wandb.init(project="context_transformer")
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (GPT2Config, GPT2LMHeadModel,AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_transformers.tokenization_gpt2 import GPT2Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', filename='log.log',
                    level=logging.DEBUG)

logger = logging.getLogger()

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
                    words = line.strip().split()[::2] + ['<eos>']
                    pos_tags = line.strip().split()[1::2] + ['<EOS>']
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
                    words = line.strip().split()[::2] + ['<eos>']
                    pos_tags = line.strip().split()[1::2] + ['<EOS>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1
                    for i,tag in enumerate(pos_tags):
                        pos_ids[pos_token] =self.pos_dictionary.word2idx[tag]
                        pos_token += 1
        return ids,pos_ids

class TaggedGPT2Corpus(object):
    def __init__(self, path,tokenizer):

        self.dictionary = Dictionary()
        self.pos_dictionary = Dictionary()
        self.tokenizer = tokenizer
        self.train = self.tokenize(os.path.join(path, 'tagged_train.txt'), tokenizer)
        self.valid = self.tokenize(os.path.join(path, 'tagged_valid.txt'), tokenizer)
        self.test = self.tokenize(os.path.join(path, 'tagged_test.txt'), tokenizer)


    def tokenize(self, path, tokenizer):
        """Tokenizes a text file into """
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                if len(line.strip().split()[::2]) > 3:
                    words = line.strip().split()[::2] + ['<eos>']
                    pos_tags = line.strip().split()[1::2] + ['<EOS>']
                    for word in words:
                        tokens += len( self.tokenizer.tokenize(word))
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
                    words = line.strip().split()[::2] + ['<eos>']
                    pos_tags = line.strip().split()[1::2] + ['<EOS>']
                    for wi,word in enumerate(words):
                        bps = self.tokenizer.tokenize(word)
                        for bp in bps:
                            ids[token] = self.tokenizer._convert_token_to_id(bp)
                            pos_ids[token] = self.pos_dictionary.word2idx[pos_tags[wi]]
                            token += 1
        return ids,pos_ids
def load_tokenize_and_batchify(tokenizer, corpus, data_dir = '../SemSynLSTM/word_language_model/data/wikitext-2/', input_len = 128):
    """
    load dataset and return train, val, test dataset

    """
    tensor_datasets = []

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="pretrained_model.")
    parser.add_argument("--model_path", type=str, help="pretrained_model.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--context_length", type=int, help="Whether to run eval on the dev set with limited context length.", default=30)
    parser.add_argument("--shuffle_pos", type=int, help="Shuffle words starting at a certain relative position.", default=30)
    parser.add_argument("--do_local_shuffle", action='store_true', help="Whether to run eval on the dev set with shuffled word order.")
    parser.add_argument("--do_global_shuffle", action='store_true', help="Whether to run eval on the dev set with shuffled word order.")
    parser.add_argument("--word_order_context_length", type=int, help="Whether to run eval on the dev set with shuffled word order.",default=None)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--data_dir', type=str, default='/home/xiongyi/dataxyz/repos/SemSynLSTM/word_language_model/data/wikitext-2/')
    parser.add_argument('--tokenized', action='store_true', help="Whether we have tokenized data ready.")
    parser.add_argument('--load_finetuned', action='store_true', help="Whether to load a finetuned model.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=10000)
    parser.add_argument('--sequence_length', type=int, default=512)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)

    args = parser.parse_args(['--output_dir', './fine_tuned_model','--do_eval', '--num_train_epochs', '1',\
                              '--model_name', 'gpt2', '--tokenized','--load_finetuned', '--context_length',\
                              '300','--shuffle_pos','200', '--do_local_shuffle'])
    #args = parser.parse_args()
    #args = parser.parse_args(['--output_dir', './tmp', '--do_eval', '--model_name', 'gpt2'])
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.warning("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.load_finetuned:
        config = GPT2Config.from_pretrained('gpt2')
        model = GPT2LMHeadModel(config)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'gpt2_1epoch.bin')))

        # tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
        model.to(device)
    elif args.model_name:
        model = GPT2LMHeadModel.from_pretrained(args.model_name)
        config = model.config

    wandb.watch(model)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Compute the max input length for the Transformer
    # Todo: Where is this used?
    sequence_length = max(config.n_ctx, args.sequence_length)
    if not args.tokenized:
        data_dir = '../SemSynLSTM/word_language_model/data/wikitext-2/' if args.data_dir is None else args.data_dir
        corpus = TaggedGPT2Corpus(data_dir, tokenizer=tokenizer)
        torch.save(corpus.train[0], 'train_id.pt')
        torch.save(corpus.train[1], 'train_pos.pt')
        torch.save(corpus.valid[0], 'val_id.pt')
        torch.save(corpus.valid[1], 'val_pos.pt')
        train_set, val_set, test_set, dictionary, pos_dictionary = load_tokenize_and_batchify(tokenizer, corpus,
                                                                                              data_dir, sequence_length)
    else:
        train_id = torch.load('/home/xiongyi/dataxyz/data/corpora/wikitext-2/train_id.pt')
        train_pos = torch.load('/home/xiongyi/dataxyz/data/corpora/wikitext-2/train_pos.pt')
        train_set = (train_id, train_pos)
        val_set = torch.load('/home/xiongyi/dataxyz/data/corpora/wikitext-2/val_id.pt')[100000:110000]
        n_batch = len(train_set[0]) // sequence_length
        input_ids = train_set[0][: n_batch * sequence_length].reshape(n_batch, sequence_length)
        pos_ids = train_set[1][: n_batch * sequence_length].reshape(n_batch, sequence_length)
        all_inputs = (input_ids, pos_ids)
        train_set=tuple (t for t in all_inputs)
    #breakpoint()
    model.to(device)
    # Prepare inputs tensors and dataloaders

    train_data = TensorDataset(*train_set)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(val_set)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=len(val_set))

    # TODO: Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    #special_tokens = ['_start_', '_delimiter_']
    #special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)

    # TODO: Add config



    # TODO: Load and encode the datasets

    logger.warning("Encoding dataset...")
    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          #max_grad_norm=args.max_grad_norm,
                          weight_decay=args.weight_decay)
                          #t_total=num_train_optimization_steps)

    if args.do_train:
        train_results = {}
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            ###eval on eval set
            # model.eval()
            # nb_eval_steps, nb_eval_examples = 0, 0
            # log_probs_sum = 0
            # perp = 0
            # average_loss = 0
            # for batch in tqdm(eval_dataloader, desc="Evaluating"):
            #     batch = tuple(t.to(device) for t in batch)
            #     input_ids, input_pos_ids = batch
            #
            #     with torch.no_grad():
            #         loss = model(input_ids, labels=input_ids)[0].detach().cpu().numpy()
            #         perp_batch = np.exp(loss)
            #         perp += perp_batch
            #         average_loss += loss
            #     nb_eval_steps += 1
            # perp /= nb_eval_steps
            # average_loss /= nb_eval_steps
            # print('loss', average_loss,'perp ', perp, 'epoch ', epoch)
            # train_results[epoch]= (perp, average_loss)

            model.train()

            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_pos_ids = batch
                loss = model(input_ids, labels=input_ids)[0]
                #breakpoint()
                #loss = args.lm_coef * losses[0] + losses[1]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} ".format(exp_average_loss)

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, args.model_name+'_epoch_' + str(args.num_train_epochs))
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        #tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned

        print (train_results)

    if args.do_eval:
        model.eval()
        with torch.no_grad():
            nb_eval_steps, nb_eval_examples = 0, 0
            perp = 0
            loss = 0
            processed_tokens = 0
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                dat = batch[0]
                #breakpoint()
                perp_batch = 0
                for i,token in enumerate(tqdm(dat)):
                    if i < args.context_length:
                        continue
                    if processed_tokens % 500 == 0 and processed_tokens:
                        print ('perp ', np.exp(loss/processed_tokens), 'processed_tokens ', processed_tokens )
                        logger.warning("'perp' = %s, 'processed_tokens = %s'", str(np.exp(loss/processed_tokens)), str(processed_tokens) )
                        wandb.log({"Eval perp": str(np.exp(loss/processed_tokens)), "Processed tokens": str(processed_tokens)})
                    input_ids = dat[i-args.context_length:i].to(device).unsqueeze(0)
                    if args.do_local_shuffle:
                        copy = input_ids[0,-args.shuffle_pos-20 : -args.shuffle_pos]
                        rand_ids = torch.randperm(len(copy))
                        copy = copy[rand_ids]
                        #random.shuffle(copy)
                        #copy.reverse()
                        input_ids[0,-args.shuffle_pos-20 : -args.shuffle_pos] = copy
                    elif args.do_global_shuffle:
                        copy = input_ids[0,:args.shuffle_pos]
                        rand_ids = torch.randperm(len(copy))
                        copy = copy[rand_ids]
                        #random.shuffle(copy)
                        #copy.reverse()
                        input_ids[0,:args.shuffle_pos]= copy

                    logits = model(input_ids)[0][0,-1,:].detach().cpu().numpy()
                    #pred_id = np.argmax(logits)
                    #pred_token = tokenizer.convert_ids_to_tokens([pred_id])[0]
                    #print (input_sent + ' ' + pred_token)
                    logprob = logits[token.item()] - logsumexp(logits)
                    #perp_tok = np.exp(-logprob)
                    #print (tokenizer.convert_ids_to_tokens([token.item()]), 'perp_tok ', perp_tok)
                    loss += -logprob
                    processed_tokens += 1
                nb_eval_steps += 1
                print ('processed ', processed_tokens)
                loss /= processed_tokens
                perp = np.exp(loss)
                # perp_word = perp / 128
                print (perp)
            result = {'eval_perp': perp}
            logger.warning("***** Eval results *****")
            logger.warning("'eval_perp' = %s", str(result['eval_perp']))
            # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            # with open(output_eval_file, "w") as writer:
            #     logger.warning("***** Eval results *****")
            #     for key in sorted(result.keys()):
            #         logger.warning("  %s = %s", key, str(result[key]))
            #         writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    main()
