import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
import logging
import argparse
import evaluate
import torch
from datasets import load_dataset, load_from_disk, load_metric
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed

nltk.download('punkt')

parser = argparse.ArgumentParser(description='Configurations for the model and training process.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
parser.add_argument('--model_path', type=str, default='google/mt5-small', help='Path to the pre-trained model')
parser.add_argument('--src_max_length', type=int, default=512, help='Maximum source sequence length')
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
wandb.init(project='nlp_proj', name=f'{args.task_name}_{args.model_path}_seed_{args.seed}')
wandb.config.update(args) 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed before initializing model.
set_seed(args.seed)

# BUG 之后得调
max_seq_len = 20

# Model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
model = model.cuda()

# Metrics
rouge_score = evaluate.load("rouge")
bleu_score = evaluate.load("bleu")

# Data
dataset = load_dataset('alt', cache_dir='data')

LANG_TOKEN_MAPPING = {
    'en': '<en>',
    'zh': '<zh>'
}

special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

def encode_input_str(text, target_lang, tokenizer, seq_len,
                     lang_token_map=LANG_TOKEN_MAPPING):
  target_lang_token = lang_token_map[target_lang]

  # Tokenize and add special tokens
  input_ids = tokenizer.encode(
      text = target_lang_token + text,
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = True,
      max_length = seq_len)

  return input_ids[0]

def encode_target_str(text, tokenizer, seq_len,
                      lang_token_map=LANG_TOKEN_MAPPING):
  token_ids = tokenizer.encode(
      text = text,
      return_tensors = 'pt',
      padding = 'max_length',
      truncation = True,
      max_length = seq_len)
  
  return token_ids[0]

def format_translation_data(translations, lang_token_map,
                            tokenizer, seq_len=128):
  # Get the languages in the translation
  langs = list(lang_token_map.keys())
  input_lang, target_lang = 'en', 'zh'

  # Get the translations for the batch
  input_text = translations[input_lang]
  target_text = translations[target_lang]

  if input_text is None or target_text is None:
    return None

  input_token_ids = encode_input_str(
      input_text, target_lang, tokenizer, seq_len, lang_token_map)
  
  target_token_ids = encode_target_str(
      target_text, tokenizer, seq_len, lang_token_map)

  return input_token_ids, target_token_ids

def transform_batch(batch):
  inputs = []
  targets = []
  for translation_set in batch['translation']:
    formatted_data = format_translation_data(
        translation_set, LANG_TOKEN_MAPPING, tokenizer, max_seq_len)
    
    if formatted_data is None:
      continue
    
    input_ids, target_ids = formatted_data
    inputs.append(input_ids.unsqueeze(0))
    targets.append(target_ids.unsqueeze(0))
    
  batch_input_ids = torch.cat(inputs).cuda()
  batch_target_ids = torch.cat(targets).cuda()

  return batch_input_ids, batch_target_ids


dataset_dict_tokenized = dataset.map(
    transform_batch,
    batched=True,
    remove_columns=dataset["train"].column_names
)
print(dataset_dict_tokenized)
assert 0

log_every = 500
eval_every = 500
save_steps = 500


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"out/trans_{args.model_path.replace('/','_')}_seed_{args.seed}",
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
    score = bleu_score.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    result["bleu"] = score["score"]
    return {k: round(v, 4) for k, v in result.items()}

# Dynamic padding in batch using a data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the trainer
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=dataset_dict_tokenized["train"],
    eval_dataset=dataset_dict_tokenized["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

logger.info("*** Start training ***")
trainer.train()