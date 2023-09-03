
import importlib
import sys
from .trainer import Trainer
import torch.distributed as dist

if __name__ == "__main__":
    module_name = sys.argv[1]
    module = importlib.import_module(f"configs.{module_name}")
    base_obj = module.Configs()
    if base_obj.DISTRIBUTED:
        dist.init_process_group(backend='nccl')
    trainer = Trainer(base_obj)
    trainer.train()
    print(f"Completed Training, and validation for {module_name}!")