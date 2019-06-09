#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:29:33 2019

@author: xiongyi
"""
import re
import numpy as np

filename = '/home/xiongyi/Codes/pytorch-pretrained-BERT/examples/zero_out_heads_seclay.log'

with open(filename ,'r') as fd:
    line = fd.readline()
    heads = []
    head_count = -1
    while (line):
        if (re.findall(r"Doing head number", line)):
            heads.append([])
            head_count += 1
        pear = re.findall(r"ALL \(average\) : Pearson = (.*),", line)
        if (len(pear) >0):
            heads[head_count].append(pear[0])            
        accs = re.findall(r"Test acc : (.*) for (.*)", line)
        if (len(accs) >0):
            heads[head_count].append(accs[0][0])
        line = fd.readline()          
        
res_mat = np.asanyarray(heads)