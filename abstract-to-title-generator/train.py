import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
import logging
import argparse
import evaluate
from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed, TrainerCallback

nltk.download('punkt')

parser = argparse.ArgumentParser(description='Configurations for the model and training process.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
parser.add_argument('--model_path', type=str, default='t5-base', help='Path to the pre-trained model')
parser.add_argument('--src_max_length', type=int, default=512, help='Maximum source sequence length')
parser.add_argument('--tgt_max_length', type=int, default=128, help='Maximum target sequence length')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
parser.add_argument('--lr', type=float, default=5.6e-5, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--use_mt5', type=bool, default=False, help='Whether to use mt5')
parser.add_argument('--resume_ckp_path', type=str, default=None, help='Resume from ckp')
parser.add_argument('--task_name', type=str, default='a2t', help='Task name')
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

# Initialize T5-base tokenizer
# BUG 这里mt5可能会有问题，待会再说吧
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Load the processed data
dataset = load_from_disk('arxiv_AI_dataset')

MAX_SOURCE_LEN = args.src_max_length
MAX_TARGET_LEN = args.tgt_max_length

def preprocess_data(example):
    
    model_inputs = tokenizer(example['abstract'], max_length=MAX_SOURCE_LEN, padding=True, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['title'], max_length=MAX_TARGET_LEN, padding=True, truncation=True)

    # Replace all pad token ids in the labels by -100 to ignore padding in the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs['labels'] = labels["input_ids"]
    return model_inputs

# Apply preprocess_data() to the whole dataset
processed_dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=['abstract', 'title'],
    desc="Running tokenizer on dataset",
)

log_every = 500
eval_every = 1000
save_steps = 1000


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"out/a2t_{args.model_path.replace('/','_')}_seed_{args.seed}_lr_{args.lr}",
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

# Initialize T5-base model
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

# Define ROGUE metrics on evaluation data
rouge_score = evaluate.load("rouge")
bleu_score = evaluate.load("bleu")
sacrebleu_score = evaluate.load("sacrebleu")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
#     # Replace -100 in the labels as we can't decode them
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
#     # ROUGE expects a newline after each sentence
#     decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
#     # Compute ROUGE scores and get the median scores
#     result = metric.compute(
#         predictions=decoded_preds, references=decoded_labels, use_stemmer=True
#     )
#     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
#     result = {k: round(v, 4) for k, v in result.items()}

#     bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]
#     result["bleu"] = bleu_score

#     return {k: round(v, 4) for k, v in result.items()}

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
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate at the first step
# Reference: https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838/6
class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

trainer.add_callback(EvaluateFirstStepCallback())

logger.info("*** Start training ***")
trainer.train()