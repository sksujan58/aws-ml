from __future__ import print_function
import logging
import sys
import argparse
import subprocess
import sys
# subprocess.check_call([sys.executable, '-m', 'conda', 'install', '-c', 'pytorch', 'pytorch==1.9.0', '-y'])
from torch import nn

import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
import json
import os
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
# import pandas as pd
# import numpy as np



def model_fn(model_dir):
#     model_path = os.path.join(model_dir, 'model.tar.gz')
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=True)
    return model, tokenizer

def predict_fn(dataset, model_and_tokenizer):
    classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] #labels
    names=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","i1","j1","k1","l1","m1","n1"] #classes_based_on_those_labels
    model, tokenizer = model_and_tokenizer
    lst=[]
    for data in dataset:
                inputs = tokenizer([data["text"]], padding=True, truncation=True, max_length=512, return_tensors="pt")
                output = model(**inputs)

                softmax_fn = nn.Softmax(dim=1)
                softmax_output = softmax_fn(output[0])
                probs=softmax_output.tolist()[0]
        
                prob_dic={}

                for i in range(len(probs)):
                    prob_dic[classes[i]]=probs[i]

                probability_list, prediction_label_list = torch.max(softmax_output, dim=1)
               
                probability = probability_list.item()

                predicted_label_idx = prediction_label_list.item()
                predicted_label = classes[predicted_label_idx]

                prediction_dict = {}
                prediction_dict['score'] = probability
                prediction_dict['Categ'] = predicted_label
                prediction_dict["pred"]  = names[predicted_label]
                prediction_dict["probabs"]=prob_dic
                prediction_dict["text"]=data["text"]
                prediction_dict["id"]  = data["id"]
                

                lst.append(prediction_dict)

    return lst
    
def output_fn(prediction_output, accept='application/jsonlines'):
    return prediction_output    
