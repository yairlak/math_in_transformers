#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:30:48 2021

@author: yl254115
"""

import argparse
import h5py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path',
                    default='../stimuli/test.txt')
parser.add_argument('-o', '--output_path',
                    default='../output/test_results.txt')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

with open(args.output_path, 'w') as fout:
    for i_seq, sequence in enumerate(open(args.input_path)):
        sequence = sequence.strip() # Remove trailing characters
        seq_length = len(sequence.split())
        
        inputs = tokenizer(sequence, return_tensors="pt")
        input_ids = inputs["input_ids"]
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0]
            
        ppl = torch.exp(neg_log_likelihood)
        line = f'{sequence} ({seq_length}): nll = {neg_log_likelihood:1.2f}, ppl={ppl:1.2f}\n'
        print(line)
        fout.write(line)
            
    
