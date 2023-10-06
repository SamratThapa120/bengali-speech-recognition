#multi-gpu training
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.run --master_port=25678 --nproc_per_node=4 train.py wav2vec2_characterwise_pretrained_ctc_preprocessor_augmented
#single-gpu training
# CUDA_VISIBLE_DEVICES=1 python train.py whisper_characterwise_nolm_ctcloss_frozenenc
