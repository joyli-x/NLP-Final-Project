from datasets import load_dataset, DatasetDict, Dataset
import json
import random
import re
import pandas as pd
import numpy as np
import torch

def clean_review(text):
    text = text.split()
    text = [x.strip() for x in text]
    text = [x.replace('\n', ' ').replace('\t', ' ') for x in text]
    text = ' '.join(text)
    text = re.sub('([.,!?()])', r' \1 ', text)
    return text
    

def get_texts(df):
    texts = 'multilabel classification: ' + df['review'].apply(clean_review)
    texts = texts.values.tolist()
    return texts


def get_labels(df):
    labels_li = [' '.join(x.lower().split()) for x in df.columns.to_list()[:8]]
    labels_matrix = np.array([labels_li] * len(df))

    mask = df.iloc[:, :8].values.astype(bool)
    labels = []
    for l, m in zip(labels_matrix, mask):
        x = l[m]
        if len(x) > 0:
            # labels.append(' , '.join(x.tolist()) + ' </s>')
            labels.append(' , '.join(x.tolist()))
        else:
            # labels.append('none </s>')
            labels.append('none')
    return labels

class T5ResDataset(torch.utils.data.Dataset):
    def __init__(self, df, indices, config, set_type=None):
        super(T5ResDataset, self).__init__()

        df = df.iloc[indices]
        self.texts = get_texts(df)
        self.set_type = set_type
        if self.set_type != 'test':
            self.labels = get_labels(df)

        self.tokenizer = config.TOKENIZER
        self.src_max_length = config.SRC_MAX_LENGTH
        self.tgt_max_length = config.TGT_MAX_LENGTH

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        src_tokenized = self.tokenizer.encode_plus(
            self.texts[index], 
            max_length=self.src_max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()

        if self.set_type != 'test':
            tgt_tokenized = self.tokenizer.encode_plus(
                self.labels[index], 
                max_length=self.tgt_max_length,
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )
            tgt_input_ids = tgt_tokenized['input_ids'].squeeze()
            tgt_attention_mask = tgt_tokenized['attention_mask'].squeeze()

            return {
                'src_input_ids': src_input_ids.long(),
                'src_attention_mask': src_attention_mask.long(),
                'tgt_input_ids': tgt_input_ids.long(),
                'tgt_attention_mask': tgt_attention_mask.long()
            }

        return {
            'src_input_ids': src_input_ids.long(),
            'src_attention_mask': src_attention_mask.long()
        }
def get_res_classification_dataset(file_path, config):
    train_df = pd.read_csv(file_path)
    label_list = [' '.join(x.lower().split()) for x in train_df.columns.to_list()[:8]]
    # train-val split
    np.random.seed(config.SEED)
    dataset_size = len(train_df)
    indices = list(range(dataset_size))
    split = int(np.floor(config.VALIDATION_SPLIT * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return T5ResDataset(train_df, train_indices, config), T5ResDataset(train_df, val_indices, config), label_list

def sample_few_shot(data, num_samples_per_class, label_key='labels'):
    # According to paper, for dataset with less than 5 labels, sample 32 examples and try to balance the number pf each class
    # For dataset with more than 5 labels, sample 8 examples from each class
    sample_per_class = {}
    few_shot_samples = []

    # Collect samples for each class
    for item in data:
        label = item[label_key]
        if label not in sample_per_class:
            sample_per_class[label] = []
        sample_per_class[label].append(item)
    
    # Sample few-shot examples for each class
    count = 0
    for label, class_exsamples in sample_per_class.items():
        sample_num = num_samples_per_class[count]
        count += 1
        
        if len(class_exsamples) <= sample_num:
            few_shot_samples.extend(class_exsamples)
        else:
            few_shot_samples.extend(random.sample(class_exsamples, sample_num))
    
    return few_shot_samples

def load_semeval_2014_task4(dataset_name, sep_token, label_shift):
    if re.match(r'laptop_*', dataset_name):
        file_path = './dataset/SemEval14-laptop/'
    elif re.match(r'restaurant_*', dataset_name):
        file_path = 'dataset/SemEval14-res/'
    else:
        raise Exception('dataset name error')
    
    dataset = {"train": [], "test": []}
    
    for split in ['train', 'test']:
        split_file_path = file_path + split + '.json'
        with open(split_file_path, 'r') as file:
            data = json.load(file)
        
        polarity_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}

        for data_id, data_item in data.items():
            polarity = polarity_mapping[data_item['polarity']] + label_shift
            text = f"{data_item['term']}{sep_token}{data_item['sentence']}"
            dataset[split].append({"text": text, "labels": polarity})

    # fs
    few_shot = False
    if dataset_name[-2:] == 'fs':
        few_shot = True
    num_samples_per_class = [10, 11, 11]  # 3类，32个样本，10,11,11
    if few_shot:
        dataset['train'] = sample_few_shot(dataset['train'], num_samples_per_class)
    
    return dataset

def load_acl_sup(label_shift, dataset_name):
    dataset = load_dataset('json', data_files={'train': 'dataset/acl-arc/train.jsonl', 'test': 'dataset/acl-arc/test.jsonl'})
    dataset = dataset.map(lambda examples: {'text': examples['text'], 'labels': examples['intent'] + label_shift})

    # fs
    few_shot = False
    if dataset_name[-2:] == 'fs':
        few_shot = True
    num_samples_per_class = [8] * 6  # 6类，每类8个样本
    if few_shot:
        dataset['train'] = sample_few_shot(dataset['train'], num_samples_per_class)

    return dataset

def load_agnews_sup(label_shift, dataset_name):
    dataset = load_dataset("ag_news", split='test')
    dataset = dataset.train_test_split(train_size=0.9, seed=2022)
    dataset = dataset.map(lambda example: {'labels': example['label'] + label_shift, 'text': example['text']})

    # fs
    few_shot = False
    if dataset_name[-2:] == 'fs':
        few_shot = True
    num_samples_per_class = [8] * 4  # 4类，每类8个样本
    if few_shot:
        dataset['train'] = sample_few_shot(dataset['train'], num_samples_per_class)

    return dataset

def get_dataset(dataset_names, sep_token):
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]

    datasets_combined = {"train": [], "test": []}
    current_label_shift = 0

    for dataset_name in dataset_names:
        if re.match(r'restaurant_*', dataset_name) or re.match(r'laptop_*', dataset_name):
            dataset = load_semeval_2014_task4(dataset_name, sep_token, current_label_shift)
            current_label_shift += 3  # To avoid label overlapping for MTL

        elif re.match(r'acl_*', dataset_name):
            dataset = load_acl_sup(current_label_shift, dataset_name)
            current_label_shift += 6  # Update the shift according to the number of classes in AG News
        
        elif re.match(r'agnews_*', dataset_name):
            dataset = load_agnews_sup(current_label_shift, dataset_name)
            current_label_shift += 4  # Update the shift according to the number of classes in AG News
            
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

        for split in ["train", "test"]:
            datasets_combined[split].extend(dataset[split])

    # Convert combined list data to Hugging Face DatasetDict
    combined_dataset = DatasetDict({
        "train": Dataset.from_dict({"text": [item["text"] for item in datasets_combined["train"]],
                                    "labels": [item["labels"] for item in datasets_combined["train"]]}),
        "test": Dataset.from_dict({"text": [item["text"] for item in datasets_combined["test"]],
                                   "labels": [item["labels"] for item in datasets_combined["test"]]})
    })

    return combined_dataset

def get_label_num(dataset_name):
    if re.match(r'restaurant_*', dataset_name) or re.match(r'laptop_*', dataset_name):
        return 3
    elif re.match(r'acl_*', dataset_name):
        return 6
    elif re.match(r'agnews_*', dataset_name):
        return 4
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
