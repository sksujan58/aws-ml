from __future__ import print_function
import logging
import sys
import argparse
import subprocess
import sys
from torch import nn
from scipy.special import softmax
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'simpletransformers'])
import json
import os
import torch
from simpletransformers.classification import ClassificationModel
import pandas as pd
import numpy as np


def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, "model.pth"))
    return model

def predict_fn(dataset, model):
    classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] #labels
    names=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","i1","j1","k1","l1","m1","n1"] #classes_based_on_those_labels
    model = model
    lst=[]
    print(dataset)
#     dataset=dataset.decode()
    dataset=json.loads(dataset)
    lst=[]
    for data in dataset:
        prediction,raw_output= model.predict([data["text"]])
        raw_output=softmax(raw_output)
        prob_dic={ str(classes[i]): str(raw_output[0][i]) for i in range(0,len(classes)) }    
        dic={}
        dic["score"]=str(raw_output[0][prediction[0]])
        dic["categ"]=str(prediction[0])
        dic["pred"]=names[prediction[0]]
        dic["probabs"]=prob_dic
        dic["text"]=data["text"]
        dic["id"]=data["id"]
        lst.append(dic)

    return lst
    
def input_fn(serialized_input_data, content_type="application/jsonlines"):
    return serialized_input_data


def output_fn(prediction_output, accept="application/jsonlines"):
    return prediction_output, accept 
