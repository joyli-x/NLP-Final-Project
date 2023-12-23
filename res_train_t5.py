import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re
import copy
from tqdm import tqdm
import gc
import wandb
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
    T5Tokenizer, 
    T5Model,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    MT5Model,
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
parser.add_argument('--use_mt5', type=bool, default=False, help='Whether to use mt5')
parser.add_argument('--resume_ckp_path', type=str, default=None, help='resume from ckp')
args = parser.parse_args()

config = ResClassificationConfig(args)

# Initialize Weights & Biases
wandb.init(project='nlp_proj', name=f'res_{args.model_path}_seed_{args.seed}')
wandb.config.update(args) 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# dataset and dataloader
train_data, val_data, label_list = get_res_classification_dataset('./data/restaurant/res_data.csv', config)
train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE)
val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

# model
class T5Model(nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()

        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        lm_labels=None
        ):

        return self.t5_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

class MT5Model(nn.Module):
    def __init__(self):
        super(MT5Model, self).__init__()

        self.t5_model = MT5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)

    def forward(
        self,
        input_ids, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        lm_labels=None
        ):

        return self.t5_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

device = config.DEVICE

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
                lm_labels=lm_labels,
                decoder_attention_mask=b_tgt_attention_mask)
            loss = outputs.loss

            val_loss += loss.item()

            # get true 
            for true_id in b_tgt_input_ids:
                true_decoded = config.TOKENIZER.decode(true_id)
                true.append(true_decoded)

            # get pred (decoder generated textual label ids)
            pred_ids = model.t5_model.generate(
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
    wandb.log(metrics)
    logger.info(metrics)
    return val_f1_score


def train(
    model,  
    train_dataloader, 
    val_dataloader, 
    criterion, 
    optimizer, 
    scheduler, 
    epoch
    ):

    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        # set model.eval() every time during training
        model.train()
        
        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)
    
        lm_labels = batch['tgt_input_ids'].to(device)
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # clear accumulated gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(input_ids=b_src_input_ids, 
                        attention_mask=b_src_attention_mask,
                        lm_labels=lm_labels,
                        decoder_attention_mask=b_tgt_attention_mask)
        loss = outputs.loss
        train_loss += loss.item()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
        # update scheduler
        scheduler.step()

    
    avg_train_loss = train_loss / len(train_dataloader)
    wandb.log({'Train Loss': avg_train_loss})
    logger.info(f'Training loss: {avg_train_loss}')

# run
def run(model):
    # setting a seed ensures reproducible results.
    # seed may affect the performance too.
    torch.manual_seed(config.SEED)

    criterion = nn.BCEWithLogitsLoss()
    
    # define the parameters to be optmized -
    # - and add regularization
    if config.FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=config.LR)

    num_training_steps = len(train_dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    max_val_micro_f1_score = float('-inf')
    for epoch in range(config.EPOCHS):
        logger.info(f'Epoch {epoch}')
        train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epoch)
        val_micro_f1_score = val(model, val_dataloader, criterion)

        if config.SAVE_BEST_ONLY:
            if val_micro_f1_score > max_val_micro_f1_score:
                best_model = copy.deepcopy(model)
                best_val_micro_f1_score = val_micro_f1_score

                model_name = f"./ckpts/res_{args.model_path.replace('/','_')}_seed_{args.seed}"
                torch.save(best_model.state_dict(), model_name + '.pt')

                logger.info(f'--- Best Model. F1 score: {max_val_micro_f1_score} -> {val_micro_f1_score}')
                max_val_micro_f1_score = val_micro_f1_score

    return best_model, best_val_micro_f1_score

if args.use_mt5:
    model = MT5Model()
else:
    model = T5Model()
model.t5_model.resize_token_embeddings(len(config.TOKENIZER))
model.to(device)

best_model, best_val_micro_f1_score = run(model)