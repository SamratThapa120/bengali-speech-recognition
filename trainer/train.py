
import importlib
import sys
import os
import numpy as np
sys.path.append("../")
import datetime
from trainer.whisper_fintune_trainer import Trainer

import torch.distributed as dist
import deepspeed

if __name__ == "__main__":

    if len(sys.argv)==3:
        module_name = sys.argv[2]
    elif len(sys.argv)==2:
        module_name = sys.argv[1]
    module = importlib.import_module(f"configs.{module_name}")
    base_obj = module.Configs()
    if base_obj.DISTRIBUTED:
        # dist.init_process_group(backend='nccl',timeout=datetime.timedelta(seconds=7200000))
        deepspeed.init_distributed()
    if base_obj.TRAIN_TYPE=="CTC":
        from trainer.whisper_fintune_ctc_trainer import Trainer
    elif base_obj.TRAIN_TYPE=="wav2vec_ctc":
        from trainer.wav2vec2_fintune_ctc_trainer import Trainer 
    elif base_obj.TRAIN_TYPE=="wav2vec_lm":
        from trainer.wav2vec2_fintune_autoreg_trainer import Trainer
    elif base_obj.TRAIN_TYPE=="whisper_deepspeed":
        from trainer.whisper_fintune_trainer_deepspeed import Trainer
    else:
        from trainer.whisper_fintune_trainer import Trainer
    trainer = Trainer(base_obj)
    trainer.train()
    print(f"Completed Training, and validation for {module_name}!")
