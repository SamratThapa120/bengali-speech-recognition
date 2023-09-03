#multi-gpu training
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE train.py whisper_characterwise
#single-gpu training
CUDA_VISIBLE_DEVICES=0 python train.py whisper_characterwise
