import os
import torch
import random
import evaluate
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass 
from time import perf_counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, disable_progress_bar
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from translation_utils import create_translation_data

dataset = load_dataset('./data/alt')
print(dataset)

print(dataset['train'][0])

# 映射函数 使用翻译字段创建一个新的字典
def transform_features(example):
    # 提取'bg'和'zh'字段并存储在新的字典中
    new_example = {'bg': example['translation']['bg'], 'zh': example['translation']['zh']}
    return new_example

# 将映射函数应用到每个样例并创建新的数据集
transformed_dataset = dataset.map(transform_features, remove_columns=['SNT.URLID', 'SNT.URLID.SNTID', 'translation', 'url'])

# 显示新数据集中的几个样例以确认是否得到了正确的数据格式
print(transformed_dataset['train'][:5])
