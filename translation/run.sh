# mt5
# CUDA_VISIBLE_DEVICES=7 python train.py --epochs=100 --seed=43
# t5
CUDA_VISIBLE_DEVICES=6 python train_v2.py --epochs=10 --lr=5e-4
# CUDA_VISIBLE_DEVICES=4 python -W ignore res_train_t5.py --epochs=40 --seed=43
# CUDA_VISIBLE_DEVICES=4 python -W ignore tmp.py --resume_ckp_path='./t5_best_model.pt'
