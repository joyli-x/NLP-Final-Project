import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
import logging
import argparse
import evaluate
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed, TrainerCallback

nltk.download('punkt')

parser = argparse.ArgumentParser(description='Configurations for the model and training process.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
parser.add_argument('--model_path', type=str, default='google/mt5-small', help='Path to the pre-trained model')
parser.add_argument('--src_max_length', type=int, default=128, help='Maximum source sequence length')
parser.add_argument('--tgt_max_length', type=int, default=128, help='Maximum target sequence length')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--resume_ckp_path', type=str, default=None, help='Resume from ckp')
parser.add_argument('--task_name', type=str, default='trans', help='Task name')
parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')
args = parser.parse_args()

# Initialize Weights & Biases
wandb.init(project='nlp_proj', name=f"{args.task_name}_{args.model_path.split('/')[-1]}_lr_{args.lr}")
wandb.config.update(args) 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed before initializing model.
set_seed(args.seed)
torch.manual_seed(args.seed)

# Model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
model = model.cuda()

# Data
# a2t
a2t_dataset = load_from_disk('./abstract-to-title-generator/arxiv_AI_dataset')

# translation
trans_data_files = {'train': './translation/data/alt/train.csv', 'validation': './translation/data/alt/val.csv'}
trans_dataset = load_dataset(
    "csv",
    delimiter=",",
    column_names=['en', 'zh'],
    data_files=trans_data_files
)

def transform_features_a2t(example):
    # Engilsh to Chinese
    new_example = {'input': 'abstract to title: ' + example['abstract'], 'output': example['title']}
    return new_example

# Process dataset
a2t_dataset = a2t_dataset.map(transform_features_a2t, remove_columns=['abstract', 'title'])

def transform_features_translation(example):
    # Engilsh to Chinese
    new_example = {'input': 'translate English to Chinese: ' + example['en'], 'output': example['zh']}
    return new_example

# Process dataset
trans_dataset = trans_dataset.map(transform_features_translation, remove_columns=['en', 'zh'])

combined_dataset_train = concatenate_datasets([trans_dataset['train'], a2t_dataset['train']])
combined_dataset_val = concatenate_datasets([trans_dataset['validation'], a2t_dataset['val']])
combined_dataset = DatasetDict({'train': combined_dataset_train, 'validation': combined_dataset_val})


def batch_tokenize_fn(examples):
    sources = examples['input']
    targets = examples['output']
    model_inputs = tokenizer(sources, max_length=args.src_max_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=args.tgt_max_length, truncation=True)

    # Replace all pad token ids in the labels by -100 to ignore padding in the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset_dict_tokenized = combined_dataset.map(
    batch_tokenize_fn,
    batched=True,
    remove_columns=combined_dataset["train"].column_names
)

log_every = 500
eval_every = 1000
save_steps = 1000


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"out/{args.task_name}_{args.model_path.split('/')[-1]}_seed_{args.seed}_lr_{args.lr}",
    evaluation_strategy="steps",
    eval_steps=eval_every,
    learning_rate=args.lr,
    load_best_model_at_end = True,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    save_steps=save_steps,
    save_total_limit=2,
    num_train_epochs=args.epochs,
    predict_with_generate=True,
    logging_steps=log_every,
    group_by_length=True,
    lr_scheduler_type=args.lr_scheduler_type,
    report_to="wandb",
)

# Define metrics on evaluation data
rouge_score = evaluate.load("rouge")
bleu_score = evaluate.load("bleu")
sacrebleu_score = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries, which is in ids into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode labels, a.k.a. reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    score = sacrebleu_score.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    result["sacrebleu"] = score["score"]
    return {k: round(v, 4) for k, v in result.items()}

# Dynamic padding in batch using a data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the trainer
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=dataset_dict_tokenized["train"],
    eval_dataset=dataset_dict_tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

trainer.add_callback(EvaluateFirstStepCallback())

logger.info("*** Start training ***")
trainer.train()