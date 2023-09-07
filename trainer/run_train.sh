#multi-gpu training
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.launch --nproc_per_node=4 train.py whisper_characterwise
#single-gpu training
# CUDA_VISIBLE_DEVICES=0 python train.py whisper_characterwise
