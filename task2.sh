#!/bin/bash

# Define the arrays of pre-trained models and datasets
MODELS=("roberta-base" "bert-base-uncased" "allenai/scibert_scivocab_uncased")
DATASETS=("restaurant_sup" "acl_sup" "agnews_sup")

# Start the outer loop for models
for model in "${MODELS[@]}"
do
  # Inner loop for datasets
  for dataset in "${DATASETS[@]}"
  do
    # Inner-most loop for random seeds
    for i in {1..5}
    do
      # Set the random seed (simply using the loop counter for simplicity)
      seed=$((42 + i))
      #echo "$model"
      
      # Run the training command with the current combination of model, dataset, and seed
      python train.py \
        --model_name_or_path="$model" \
        --dataset_name="$dataset" \
        --sep_token="[SEP]" \
        --output_dir="./output/" \
        --do_train="True" \
        --do_eval="True" \
        --per_device_train_batch_size=16 \
        --per_device_eval_batch_size=16 \
        --num_train_epochs=25 \
        --seed="$seed" \
        --report_to="wandb" \
        --run_name="seed$seed" \
        --logging_steps=216 \
        --evaluation_strategy="epoch"
    done
  done
done
