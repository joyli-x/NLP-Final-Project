# mt5
CUDA_VISIBLE_DEVICES=4 python -W ignore res_train_t5.py --epochs=100 --seed=43 --use_mt5=True --model_path='google/mt5-small' --lr=7e-4
# t5
# CUDA_VISIBLE_DEVICES=4 python -W ignore res_train_t5.py --epochs=40 --seed=43
# CUDA_VISIBLE_DEVICES=4 python -W ignore tmp.py --resume_ckp_path='./t5_best_model.pt'
