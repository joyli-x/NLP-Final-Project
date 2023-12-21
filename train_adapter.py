import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.adapters import AdapterArguments, AdapterTrainer, AutoAdapterModel, setup_adapter_training
from transformers.trainer_utils import get_last_checkpoint
from dataHelper import get_dataset, get_label_num
from adapters.adapters import load_adapter_model


logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    dataset_name: Optional[str] = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    num_labels: Optional[int] = field(
        default = None, metadata={"help": "The number of labels in the dataset."}
    )
    sep_token: Optional[str] = field(
        default = None, metadata={"help": "The separator tokens in the dataset."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3."}
    )
    peft_model: Optional[str] = field(
        default=None, metadata={"help": "The name of the peft model. It can be chosen from ['lora, 'adapter']"}
    )
    r: Optional[int] = field(
        default=8, metadata={"help": "Lora parameter r."}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "Lora parameter alpha."}
    )

'''
	initialize logging, seed, argparse...
'''
# init args
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
if model_args.peft_model=='adapter':
    logger.info('adapter model')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdapterArguments))
    model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

if model_args.model_name_or_path=="allenai/scibert_scivocab_uncased":
    os.environ["WANDB_PROJECT"] = 'scibert_scivocab_uncased' + '_' + data_args.dataset_name
elif model_args.model_name_or_path=="openlm-research/open_llama_3b":
    os.environ["WANDB_PROJECT"] = model_args.peft_model + '_' + data_args.dataset_name
else:
    os.environ["WANDB_PROJECT"] = model_args.model_name_or_path + '_' + data_args.dataset_name

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger.setLevel(logging.DEBUG)

# Log on each process the small summary:
logger.info(f"Data parameters: {data_args}")
logger.info(f"Model parameters: {model_args}")
logger.info(f"Training parameters {training_args}")

# Set seed before initializing model.
set_seed(training_args.seed)

'''
load datasets
'''
# Load dataset
dataset = get_dataset(data_args.dataset_name, data_args.sep_token)
data_args.num_labels = get_label_num(data_args.dataset_name)

'''
load models
'''
model_args.cache_dir = './model/' + model_args.model_name_or_path + '_' + data_args.dataset_name + '/'
logger.info(f"model_args.cache_dir: {model_args.cache_dir}")
# Load pretrained model and tokenizer
config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    num_labels=data_args.num_labels,
    cache_dir=model_args.cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    cache_dir=model_args.cache_dir,
)

if model_args.peft_model == "lora":
    logger.info('lora model')
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=model.args.r, lora_alpha=model.args.lora_alpha, lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

elif model_args.peft_model == "adapter":
    model = AutoAdapterModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    # task_name写什么呢？
    model.add_classification_head(data_args.dataset_name, num_labels=data_args.num_labels)


'''
process datasets and build up datacollator
'''

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Get the metric function
f1_metric = evaluate.load('./model/evaluate/metrics/f1')
accuracy_metric = evaluate.load("./model/evaluate/metrics/accuracy")
logger.info('metric loaded')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # print(predictions.shape, type(predictions))
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    micro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="micro")
    macro_f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {"Accuracy": accuracy, "Micro F1": micro_f1, "Macro F1": macro_f1}

# Data collator
# BUG 这里已经map了还需要DataCollatorWithPadding吗？
# DataCollatorWithPadding是用于做padding的，和用tokenizer不一样
# 如果tokenizer里面写了padding和max_pad_length,就直接用default_data_collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize our Trainer
# Setup adapters
setup_adapter_training(model, adapter_args, data_args.dataset_name)
# Initialize our Trainer
trainer_class = AdapterTrainer if adapter_args.peft_model=='adapter' else Trainer
# 其实这里传了tokenizer就没必要再指定data_collator了
trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

# Training
logger.info("*** Start training ***")
train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)

# Test
# BUG 这里应该用evaluate还是predict？
# logger.info("*** Test ***")
# metrics = trainer.evaluate(eval_dataset=test_dataset)
# trainer.log_metrics("eval", metrics)
