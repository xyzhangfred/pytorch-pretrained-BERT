#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:29:33 2019

@author: xiongyi
"""
import re
import numpy as np

filename = '/home/xiongyi/Codes/pytorch-pretrained-BERT/examples/11_06_00_02DeBERT_root.log'
res = []
with open(filename ,'r') as fd:
    line = fd.readline()    
    while (line):
        pear = re.findall(r"Pearson = (.*),", line)
        #print (pear)
        if (len(pear) >0):
            res.append(pear[0])            
        accs = re.findall(r"Test acc : (.*) for (.*)", line)
        if (len(accs) >0):
            #res.append(accs[0][1] + ':' + accs[0][0])
            res.append(accs[0][0])
        line = fd.readline()          
tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                  'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                  'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                  'Length', 'WordContent', 'Depth', 'TopConstituents',
                  'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                  'OddManOut', 'CoordinationInversion']
#res_mat = np.asanyarray(heads)
