
import importlib
import sys
import os
sys.path.append("../")

from trainer.whisper_fintune_trainer import Trainer

import torch.distributed as dist

if __name__ == "__main__":
    if len(sys.argv)==3:
        module_name = sys.argv[2]
    elif len(sys.argv)==2:
        module_name = sys.argv[1]
    module = importlib.import_module(f"configs.{module_name}")
    base_obj = module.Configs()
    if base_obj.DISTRIBUTED:
        dist.init_process_group(backend='nccl')
    if base_obj.TRAIN_TYPE=="CTC":
        from trainer.whisper_fintune_ctc_trainer import Trainer
    else:
        from trainer.whisper_fintune_trainer import Trainer
    trainer = Trainer(base_obj)
    trainer.train()
    print(f"Completed Training, and validation for {module_name}!")