from __future__ import print_function
import logging
import sys
import argparse
import subprocess
import sys
# subprocess.check_call([sys.executable, '-m', 'conda', 'install', '-c', 'pytorch', 'pytorch==1.9.0', '-y'])
from torch import nn
from scipy.special import softmax
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'datasets'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'simpletransformers'])
import json
import os
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from simpletransformers.classification import ClassificationModel
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
os.makedirs("./results",exist_ok=True)


def load_train_data(file_path):
    # Take the set of files and read them all into a single pandas dataframe
    df = pd.read_csv(os.path.join(file_path, "train.csv")) 
    df=df.dropna()          
    return df


def load_test_data(file_path):
    # Take the set of files and read them all into a single pandas dataframe
    df = pd.read_csv(os.path.join(file_path, "test.csv"))
    df=df.dropna()
    return df        
    

def _parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--model_id", type=str)
    
    
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()

if __name__ == '__main__':
    
    args, unknown = _parse_args()
    model_name = args.model_id
    
    train_data = load_train_data(args.train)
    eval_data= load_test_data(args.test)
    
    train_args ={
         "num_train_epochs": args.epochs,"do_lower_case": True,
         "train_batch_size":8,"overwrite_output_dir":True,"evaluate_during_training":True}
    
    model =ClassificationModel("bert",model_name,num_labels=20,args=train_args)
    model.train_model(train_data,eval_df=eval_data)
    ml=ClassificationModel("bert","outputs/best_model",num_labels=20,use_cuda=False)
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(ml, path)
    
    inference_path = os.path.join(args.model_dir)
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))
