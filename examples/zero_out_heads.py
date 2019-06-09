#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:42:34 2019

@author: xiongyi
"""
from __future__ import absolute_import, division, unicode_literals

import sys,os
import io
import numpy as np
import logging
from logging.handlers import WatchedFileHandler
logging.basicConfig(filename = 'zero_out_heads_seclay.log',format='%(asctime)s : %(message)s', level=logging.DEBUG)


import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import argparse

from extract_features import convert_examples_to_features, InputExample
# Set PATHs
PATH_TO_SENTEVAL = '/home/xiongyi/Codes/SentEval/examples'
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL,'../data')
model = None
tokenizer = None
device = None
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set up logger
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32,
                             'tenacity': 3, 'epoch_size': 2}

#
#def extract_zvec(model, device, batch, tokenizer, seq_length = 64, layer_no = -1):
#    examples = []
#    unique_id = 0
#    print ('batch size ', len(batch))
#    for sent in batch:
#        sent = sent.strip()
#        text_b = None
#        text_a = sent
#        examples.append(
#            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
#        unique_id += 1
#    #print ('len(batch)', len(batch))
#    features = convert_examples_to_features(examples, seq_length, tokenizer)
#
#    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#    #print ('all_input_ids.shape', all_input_ids.shape)
#    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
#    #print ('all_input_ids.shape', all_input_ids.shape)
#    #all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
#    #print ('all_example_index.shape', all_example_index.shape)
#    #print ('all_input_ids.shape', all_input_ids.shape)
#
#
#    model.eval()
#    #for input_ids, input_mask, example_indices in zip(all_input_ids,all_input_mask,all_example_index):
#    #    input_ids = input_ids.to(device)
#    #    input_mask = input_mask.to(device)
#    print ('all_input_ids.shape', all_input_ids.shape)
#    print ('all_input_mask.shape', all_input_mask.shape)
#    debug_memory()
#    all_encoder_layers, _ = model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
#    
#    print ('finished a batch \n\n')
#    return all_encoder_layers[layer_no]


def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in tensors.items():
        print('{}\t{}'.format(*line))

# SentEval prepare and batcher
def prepare(params, samples):
    params.batch_size = 128
    return

def batcher(params, batch):
    #print ('batch size' ,len(batch))
    batch = [sent if sent != [] else ['.'] for sent in batch]
    batch = [' '.join(sent) for sent in batch]
    #print ('batch', batch)
    examples = []
    unique_id = 0
    #print ('batch size ', len(batch))
    for sent in batch:
        sent = sent.strip()
        text_b = None
        text_a = sent
        examples.append(
            InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
        unique_id += 1

    features = convert_examples_to_features(examples, params['bert'].seq_length, tokenizer)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    
    all_encoder_layers, _ = params['bert'](all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
   
    
    ###get z_vec
    #get the output of previous layer
    prev_out = all_encoder_layers[params['bert'].layer_no -1]
    
    #print ('here')
    #print ('prev_out.shape ', prev_out.shape)
    #print ('all_input_mask.shape ', all_input_mask.shape)
    ##apply self-attention to it
    
    
    extended_attention_mask = all_input_mask.cuda().unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(params['bert'].parameters()).dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    #print ('here?')
    embeddings = next(params['bert'].children()).encoder.layer[params['bert'].layer_no].attention.self(prev_out, extended_attention_mask)
    ##do mean/max pooling
    
    #print ('befor shape', embeddings.shape)
    
    if params['bert'].head_no is not None:
        if params['bert'].head_no != 'random':
            embeddings[:,:,:64 * params['bert'].head_no] = 0
            embeddings[:,:,64 * (params['bert'].head_no +1):] = 0
        else:
           
            embeddings[:,:,params['bert'].notrandidx] = 0
    #do the following calculation
    attention_output = next(params['bert'].children()).encoder.layer[params['bert'].layer_no].attention.output(embeddings, prev_out)
    intermediate_output = next(params['bert'].children()).encoder.layer[params['bert'].layer_no].intermediate(attention_output)
    layer_output = next(params['bert'].children()).encoder.layer[params['bert'].layer_no].output(intermediate_output, attention_output)
    
    final_embeddings = layer_output.detach().mean(1).cpu().numpy()
    
    #print ('after shape', embeddings.shape)

    #print ('embeddings.shape ', embeddings.shape)
    #print ('finished a batch \n\n')

    return final_embeddings


def main(head_no = None):

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--layer_no", default= -2, type=int)
    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for predictions.")

    args = parser.parse_args()
    global model, tokenizer, device

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info('\nDoing head number '+str(head_no) + '!\n')
    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    params_senteval['bert'] = model
    params_senteval['bert'].layer_no = args.layer_no
    params_senteval['bert'].seq_length = args.max_seq_length
    params_senteval['bert'].head_no = head_no
    params_senteval['bert'].randidx = np.random.choice(np.arange(768), size = 64, replace=False)
    params_senteval['bert'].notrandidx = [i for i in range(768) if i not in params_senteval['bert'].randidx] 
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12','SICKEntailment','Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
    
    

if __name__ == "__main__":
    main()
    for head_no in range(12):
        main(head_no)

    main('random')

