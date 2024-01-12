# Can we teach a model twice?
## Background
In the industrial sector, to meet specific requirements, developers often fine-tune pre-trained models based on open source or previously trained models. This fine-tuning process adapts the pre-trained model to specific downstream tasks. When facing new requirements, developers encounter the following issue: should they use a previously fine-tuned model for old requirements or start training from scratch using a pre-trained model that has not undergone fine-tuning?

## Experiment
We use three tasks (abstract to title, translation, review classification) to explore whether a model can be taught twice. Then we go a step further to explore whether it can learn various tasks simultaneously. More details can be found in ```report.pdf```.

## Data collection
For restaurant review classification, the dataset is in ```./res_classificatioin/data```. For abstract to title, we use the [cs.AI of arXiv dataset](https://github.com/nerdimite/abstract-to-title-generator/tree/main/arxiv_AI_dataset). For translation, we use the [alt dataset](https://huggingface.co/datasets/alt).

## Quick start
First get your environment ready:
```
conda create -n test_env python=3.9.0
conda activate test_env
pip install -r requirements.txt
```

To reproduce the results on the three task, you can go to the coresponding folder and run the following command:
```
./run.sh
```
Then you can test your model on the test set:
```
python eval.py
```
For translation and abstract to title, we sample 10 entries from the test set and show the results in ```res.txt```. 

## Our conclusion
- Models can be taught twice. 
- Models can simultaneously learn multiple tasks, but its performance will be lower than directly finetuning on specific task. When encountering new tasks, they have the potential to exhibit improved performance and require less time for finetuning.
- For a pretrained language model, instruction-tuning + finetune on specific task > instruction-tuning + finetune twice > finetune twice >= finetune once

