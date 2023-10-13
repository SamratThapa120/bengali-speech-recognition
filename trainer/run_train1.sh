#multi-gpu training
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run --master_port=25678 --nproc_per_node=4 train.py whisper_large_alldataset_augment
#single-gpu training
# CUDA_VISIBLE_DEVICES=1 python train.py whisper_characterwise_nolm_ctcloss_frozenenc
