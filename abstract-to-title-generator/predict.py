import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

original_model_path = 't5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(original_model_path).to('cuda')
original_tokenizer = AutoTokenizer.from_pretrained(original_model_path)

model_path = "/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/a2t_t5-base_seed_43/checkpoint-45000"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

model_path2 = "/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/checkpoint-21000"
model2 = AutoModelForSeq2SeqLM.from_pretrained(model_path2).to('cuda')
tokenizer2 = AutoTokenizer.from_pretrained(model_path2)

temperature = 0.9
num_beams = 4
max_gen_length = 128

# abstract = """In this paper, we question if self-supervised learning provides
# new properties to Vision Transformer (ViT) [19] that
# stand out compared to convolutional networks (convnets).
# Beyond the fact that adapting self-supervised methods to this
# architecture works particularly well, we make the following
# observations: first, self-supervised ViT features contain
# explicit information about the semantic segmentation of an
# image, which does not emerge as clearly with supervised
# ViTs, nor with convnets. Second, these features are also excellent
# k-NN classifiers, reaching 78.3% top-1 on ImageNet
# with a small ViT. Our study also underlines the importance of
# momentum encoder [33], multi-crop training [10], and the
# use of small patches with ViTs. We implement our findings
# into a simple self-supervised method, called DINO, which
# we interpret as a form of self-distillation with no labels.
# We show the synergy between DINO and ViTs by achieving
# 80.1% top-1 on ImageNet in linear evaluation with ViT-Base"""

# Load the processed data
dataset = load_from_disk('arxiv_AI_dataset')

with open('res.txt', 'w') as f:
    for i in range(5, 10):
        abstract = dataset['test'][i]['abstract']
        original_title = dataset['test'][i]['title']
        print(f'case{i}: ', file=f)
        print(f"abstract: {abstract}", file=f)
        print(f"original title: {original_title}", file=f)

        inputs = tokenizer([abstract], max_length=512, return_tensors='pt')

        # original
        title_ids = original_model.generate(
            inputs['input_ids'].to('cuda'), 
            num_beams=num_beams, 
            temperature=temperature, 
            max_length=max_gen_length,
            do_sample=True, 
            early_stopping=True
        )
        title = original_tokenizer.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"original model title: {title}", file=f)

        # finetune
        title_ids = model2.generate(
            inputs['input_ids'].to('cuda'), 
            num_beams=num_beams, 
            temperature=temperature, 
            max_length=max_gen_length, 
            do_sample=True,
            early_stopping=True
        )
        title = tokenizer2.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"21000 finetuned model title: {title}", file=f)

        # finetune
        title_ids = model.generate(
            inputs['input_ids'].to('cuda'), 
            num_beams=num_beams, 
            temperature=temperature, 
            max_length=max_gen_length, 
            do_sample=True,
            early_stopping=True
        )
        title = tokenizer.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"45000 finetuned model title: {title}", file=f)

        print('\n', file=f)