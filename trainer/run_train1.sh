#multi-gpu training
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.run --master_port=25678 --nproc_per_node=2 train.py whisper_characterwise_pretrained
#single-gpu training
# CUDA_VISIBLE_DEVICES=1 python train.py whisper_characterwise_nolm_ctcloss_frozenenc
