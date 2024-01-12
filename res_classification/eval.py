import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import re
import copy
import os
from tqdm import tqdm
# import wandb
import logging
import argparse
from utils import ResClassificationConfig, get_ohe
from dataHelper import get_res_classification_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score, 
    recall_score,
    f1_score, 
    classification_report
)

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    get_linear_schedule_with_warmup
)

parser = argparse.ArgumentParser(description='Configurations for the model and training process.')

parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
parser.add_argument('--model_path', type=str, default='t5-base', help='Path to the pre-trained model')
parser.add_argument('--src_max_length', type=int, default=450, help='Maximum source sequence length')
parser.add_argument('--tgt_max_length', type=int, default=20, help='Maximum target sequence length')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
parser.add_argument('--validation_split', type=float, default=0.25, help='Fraction of the data to use as validation')
parser.add_argument('--full_finetuning', type=bool, default=True, help='Whether to fine tune the entire model or just the head')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use for training')
parser.add_argument('--save_best_only', type=bool, default=True, help='Whether to save only the best model')
parser.add_argument('--n_validate_dur_train', type=int, default=3, help='Number of validation runs during training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--resume_ckp_path', type=str, default=None, help='resume from ckp')
parser.add_argument('--task_name', type=str, default='res', help='task name')
args = parser.parse_args()

config = ResClassificationConfig(args)

# Initialize Weights & Biases
# wandb.init(project='nlp_proj', name=f"{args.task_name}_{args.model_path.split('/')[-1]}_lr_{args.lr}")
# wandb.config.update(args) 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# dataset and dataloader
train_data, val_data, test_data, label_list = get_res_classification_dataset('./data/restaurant/res_data.csv', config)
train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE)
val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE)

device = config.DEVICE

# model
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
model.resize_token_embeddings(len(config.TOKENIZER))
logger.info(f'**** model ****: {args.model_path}')

model.to(device)

# engine

def val(model, val_dataloader, criterion):
    
    val_loss = 0
    true, pred = [], []
    
    # set model.eval() every time during evaluation
    model.eval()
    
    for step, batch in enumerate(val_dataloader):
        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)
    
        b_tgt_input_ids = batch['tgt_input_ids']
        lm_labels = b_tgt_input_ids.to(device)
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # using torch.no_grad() during validation/inference is faster -
        # - since it does not update gradients.
        with torch.no_grad():
            # forward pass
            outputs = model(
                input_ids=b_src_input_ids, 
                attention_mask=b_src_attention_mask,
                labels=lm_labels,
                decoder_attention_mask=b_tgt_attention_mask)
            loss = outputs.loss

            val_loss += loss.item()

            # get true 
            for true_id in b_tgt_input_ids:
                true_decoded = config.TOKENIZER.decode(true_id)
                true.append(true_decoded)

            # get pred (decoder generated textual label ids)
            pred_ids = model.generate(
                input_ids=b_src_input_ids, 
                attention_mask=b_src_attention_mask
            )
            pred_ids = pred_ids.cpu().numpy()
            for pred_id in pred_ids:
                pred_decoded = config.TOKENIZER.decode(pred_id)
                pred.append(pred_decoded)

    true_ohe = get_ohe(true, label_list)
    pred_ohe = get_ohe(pred, label_list)

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(true_ohe, pred_ohe)
    val_f1_score = f1_score(true_ohe, pred_ohe, average='samples')
    val_recall_score = recall_score(true_ohe, pred_ohe, average='samples')
    val_precision_score = precision_score(true_ohe, pred_ohe, average='samples')

    metrics = {
        "Val Loss": avg_val_loss,
        "Val Accuracy": val_accuracy,
        "Val F1 Score": val_f1_score,
        "Val Recall Score": val_recall_score,
        "Val Precision Score": val_precision_score
    }
    logger.info(metrics)
    return val_f1_score

val(model, test_dataloader, criterion=None)
