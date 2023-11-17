from __future__ import print_function
import logging
import sys
import argparse
import subprocess
import sys
# subprocess.check_call([sys.executable, '-m', 'conda', 'install', '-c', 'pytorch', 'pytorch==1.9.0', '-y'])
from torch import nn

import subprocess
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'datasets'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers'])
import json
import os
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
os.makedirs("./results",exist_ok=True)


class PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

def load_train_data(file_path):
    # Take the set of files and read them all into a single pandas dataframe
    df = pd.read_csv(os.path.join(file_path, "train.csv")) 
    df=df.dropna()
    texts = list(df["documents"])
    labels = df["labels"].values
    train_encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    train_dataset = PytorchDataset(train_encodings,labels)            
    return train_dataset


def load_test_data(file_path):
    # Take the set of files and read them all into a single pandas dataframe
    df = pd.read_csv(os.path.join(file_path, "test.csv"))
    df=df.dropna()
    texts = list(df["documents"])
    labels = df["labels"].values
    test_encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    test_dataset = PytorchDataset(test_encodings,labels) 
    return test_dataset    
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }    
    

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
# max sequence length for each document/sentence sample
    max_length = 512
# load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=20)
    
    train_data = load_train_data(args.train)
    eval_data= load_test_data(args.test)
    

    
    training_args = TrainingArguments(
    output_dir="./results",          # output directory
    num_train_epochs=args.epochs,              # total number of training epochs
    per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.train_batch_size,   # batch size for evaluation 
    weight_decay=0.01,               # strength of weight decay      
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    evaluation_strategy="epoch", 
    save_strategy="epoch" ,
    
     )
    
    trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_data,         # training dataset
    eval_dataset=eval_data,          # evaluation dataset
    tokenizer=tokenizer,    
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
    )
    
    trainer.train()
    eval_result = trainer.evaluate(eval_dataset=eval_data)
    
    trainer.save_model(args.model_dir)
#     tokenizer.save_pretrained(args.model_dir)
    inference_path = os.path.join(args.model_dir, "code/")
    os.makedirs(inference_path, exist_ok=True)
    os.system("cp inference.py {}".format(inference_path))