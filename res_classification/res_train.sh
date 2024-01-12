# mt5
# CUDA_VISIBLE_DEVICES=1 python -W ignore res_train_t5.py --epochs=100 --seed=44 --model_path='google/mt5-small' --lr=7e-4
# res mt5
# CUDA_VISIBLE_DEVICES=1 python res_train_t5.py --epochs=100 --task_name='trans-res' --model_path='../translation/out/trans_google_mt5-small_seed_42_lr_0.0005/checkpoint-11000' --lr=7e-4 --seed=43
# CUDA_VISIBLE_DEVICES=1 python res_train_t5.py --epochs=100 --task_name='trans-res' --model_path='../translation/out/trans_google_mt5-small_seed_42_lr_0.0005/checkpoint-11000' --lr=7e-4 --seed=44
# res t5

# a2t-res mt5
# CUDA_VISIBLE_DEVICES=1 python res_train_t5.py --epochs=100 --task_name='a2t&trans-res-mt5' --model_path='/DATA1/xuechang/lzy/NLP-Final-Project/out/a2t&trans_mt5-small_seed_42_lr_0.0005/checkpoint-33500' --lr=7e-4

# t5/flant5
# CUDA_VISIBLE_DEVICES=7 python res_train_t5.py --epochs=40 --model_path='/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/a2t_flan-t5-small_seed_42_lr_0.0001/checkpoint-18000' --lr=1e-4 --task_name='a2t-res-flan-t5-small'

# test
CUDA_VISIBLE_DEVICES=7 python -W ignore eval.py --model_path='/DATA1/xuechang/lzy/NLP-Final-Project/res_classification/out/res_mt5-small_seed_42_lr_0.0007'


