# mt5
# CUDA_VISIBLE_DEVICES=7 python train.py --epochs=100 --seed=43
# t5
# CUDA_VISIBLE_DEVICES=3 python train_v2.py --epochs=10 --lr=5e-4 --seed=43
# CUDA_VISIBLE_DEVICES=6 python train_v2.py --epochs=10 --lr=5e-4 --seed=44
# CUDA_VISIBLE_DEVICES=4 python -W ignore res_train_t5.py --epochs=40 --seed=43

# res-trans
# CUDA_VISIBLE_DEVICES=3 python train_v2.py --task_name='res-trans' --epochs=10 --lr=5e-4 --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/res_classification/out/res_mt5-small_seed_42_lr_0.0007"
# a2t-trans
# CUDA_VISIBLE_DEVICES=3 python train_v2.py --task_name='a2t-trans-mt5' --epochs=10 --lr=5e-4 --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/abstract-to-title-generator/out/a2t_mt5-small_seed_42_lr_0.0005/checkpoint-22000"

# # eval
CUDA_VISIBLE_DEVICES=3 python eval.py --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/translation/out/trans_google_mt5-small_seed_42_lr_0.0005/checkpoint-11000"
# CUDA_VISIBLE_DEVICES=3 python eval.py --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/translation/out/res-trans_res_mt5-small_seed_42_lr_0.0007_seed_43_lr_0.0005/checkpoint-10500"
# CUDA_VISIBLE_DEVICES=3 python eval.py --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/translation/out/a2t-trans-mt5_checkpoint-22000_seed_42_lr_0.0005/checkpoint-10000"
# CUDA_VISIBLE_DEVICES=3 python eval.py --model_path="/DATA1/xuechang/lzy/NLP-Final-Project/out/a2t&trans_mt5-small_seed_42_lr_0.0005/checkpoint-23500"


