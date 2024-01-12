# mt5
# CUDA_VISIBLE_DEVICES=7 python train.py --epochs=100 --seed=43
# t5
# CUDA_VISIBLE_DEVICES=6 python train.py --epochs=10 --lr=5e-4 --task_name='res-a2t-flan-t5-small' --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/res_classification/out/res_flan-t5-small_seed_42_lr_0.0001"
# CUDA_VISIBLE_DEVICES=5 python train.py --epochs=10 --lr=1e-4 --task_name='res-a2t-flant5' --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/res_classification/out/res_flan-t5-base_seed_42_lr_0.0001" --batch_size=8

# t5/mt5 res-a2t
# CUDA_VISIBLE_DEVICES=5 python train.py --epochs=10 --seed=42 --lr=5e-4 --task_name='res-a2t' --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/res_classification/out/res_mt5-small_seed_42_lr_0.0007"
# CUDA_VISIBLE_DEVICES=5 python train.py --epochs=10 --seed=42 --lr=7e-4 --task_name='res-a2t' --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/res_classification/out/res_mt5-small_seed_42_lr_0.0007"
# CUDA_VISIBLE_DEVICES=5 python train.py --epochs=10 --seed=44 --lr=1e-4 --task_name='res-a2t' --model_path="../res_classification/out/res_t5-base_seed_44_lr_0.0001"

# mt5 trans-a2t
# CUDA_VISIBLE_DEVICES=6 python train.py --epochs=10 --seed=42 --lr=5e-4 --task_name='trans-a2t-mt5' --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/translation/out/trans_google_mt5-small_seed_42_lr_0.0005/checkpoint-11000"

# eval
CUDA_VISIBLE_DEVICES=6 python -W ignore eval.py --seed=42 --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/a2t-flan-t5-small_flan-t5-small_seed_42_lr_0.0005/checkpoint-9000"
# CUDA_VISIBLE_DEVICES=6 python -W ignore eval.py --seed=42 --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/a2t_mt5-small_seed_42_lr_0.0005/checkpoint-15000"
# CUDA_VISIBLE_DEVICES=6 python -W ignore eval.py --seed=42 --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/res-a2t_res_mt5-small_seed_42_lr_0.0007_seed_42_lr_0.0005/checkpoint-15000"
# CUDA_VISIBLE_DEVICES=6 python -W ignore eval.py --seed=42 --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/trans-a2t-mt5_checkpoint-11000_seed_42_lr_0.0005/checkpoint-9000"
# CUDA_VISIBLE_DEVICES=6 python -W ignore eval.py --seed=42 --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/out/a2t&trans_mt5-small_seed_42_lr_0.0005/checkpoint-23500"


