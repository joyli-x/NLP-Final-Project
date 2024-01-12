import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
from datasets import load_from_disk, load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

original_model_path = 'google/mt5-small'
original_model = AutoModelForSeq2SeqLM.from_pretrained(original_model_path).to('cuda')
original_tokenizer = AutoTokenizer.from_pretrained(original_model_path)

model_path = "/DATA1/xuechang/lzy/NLP-Final-Project/translation/out/trans_google_mt5-small_seed_42_lr_0.0005/checkpoint-10500"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_path)

model_path2 = "/DATA1/xuechang/lzy/NLP-Final-Project/translation/out/trans_google_mt5-small_seed_42_lr_0.001/checkpoint-6500"
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
data_files = {'test': './data/alt/test.csv'}
dataset = load_dataset(
    "csv",
    delimiter=",",
    column_names=['en', 'zh'],
    data_files=data_files
)

# 在1,1000中随机20个数
random_list = np.random.randint(1, 1000, size=20)

with open('res.txt', 'w') as f:
    for i in random_list:
        abstract = dataset['test'][int(i)]['en']
        original_title = dataset['test'][int(i)]['zh']
        print(f'case{i}: ', file=f)
        print(f"en: {abstract}", file=f)
        print(f"gt translation: {original_title}", file=f)

        inputs = tokenizer([abstract], max_length=512, return_tensors='pt')

        # original
        title_ids = original_model.generate(
            inputs['input_ids'].to('cuda'), 
            num_beams=num_beams, 
            temperature=temperature, 
            max_length=max_gen_length,
            do_sample=True, 
            early_stopping=True,
            # truncation=True,
        )
        title = original_tokenizer.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"original model translation: {title}", file=f)

        # finetune
        title_ids = model2.generate(
            inputs['input_ids'].to('cuda'), 
            num_beams=num_beams, 
            temperature=temperature, 
            max_length=max_gen_length, 
            do_sample=True,
            early_stopping=True,
            # truncation=True,
        )
        title = tokenizer2.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"11000 finetuned model translation: {title}", file=f)

        # finetune
        title_ids = model.generate(
            inputs['input_ids'].to('cuda'), 
            num_beams=num_beams, 
            temperature=temperature, 
            max_length=max_gen_length, 
            do_sample=True,
            early_stopping=True,
            # truncation=True,
        )
        title = tokenizer.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"45000 finetuned model translation: {title}", file=f)

        print('\n', file=f)