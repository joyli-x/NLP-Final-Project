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
    get_linear_schedule_with_warmup
)

class ResClassificationConfig:
    def __init__(self, args):
        super(ResClassificationConfig, self).__init__()

        self.SEED = args.seed
        self.MODEL_PATH = args.model_path

        # data
        self.TOKENIZER = T5Tokenizer.from_pretrained(self.MODEL_PATH)
        self.SRC_MAX_LENGTH = args.src_max_length
        self.TGT_MAX_LENGTH = args.tgt_max_length
        self.BATCH_SIZE = args.batch_size
        self.VALIDATION_SPLIT = args.validation_split

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = args.full_finetuning
        self.LR = args.lr
        self.OPTIMIZER = args.optimizer
        self.CRITERION = 'BCEWithLogitsLoss'
        self.SAVE_BEST_ONLY = args.save_best_only
        self.N_VALIDATE_DUR_TRAIN = args.n_validate_dur_train
        self.EPOCHS = args.epochs

def get_ohe(x, label_list):
    labels_li_indices = dict()
    for idx, label in enumerate(label_list):
        labels_li_indices[label] = idx

    ohe = []
    for res in x:
        temp = [0]*8
        for label in label_list:
            if label in res:
                temp[labels_li_indices[label]] = 1
        ohe.append(temp)
    ohe = np.array(ohe)
    return ohe
