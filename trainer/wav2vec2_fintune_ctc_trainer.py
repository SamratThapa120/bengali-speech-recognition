import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .utils import setup_logger,MetricsStore
import os
from tqdm import tqdm
from bengali_asr.callbacks.evaluation import ModelValidationCallback
from bengali_asr.callbacks.examples_test_evaluation import LongFormatExamplesEvaluation
from contextlib import nullcontext
class Trainer:
    def unfreeze(self,epoch):
        for name, params in self.model.named_parameters():
            params.requires_grad = True
            params.requires_grad_(True)
        if epoch in self.UNFREEZE_EPOCH:
            unf_name = self.UNFREEZE_NAME[self.UNFREEZE_EPOCH.index(epoch)]
            print("Freezing parameters before: ",unf_name)
            freeze= True
            for name, params in self.model.named_parameters():
                if unf_name in name:
                    freeze=False
                if freeze:
                    params.requires_grad = False
                    params.requires_grad_(False)
    def __init__(self, base_obj):

        self.__dict__.update(base_obj.get_all_attributes())

        if self.DISTRIBUTED:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"
            self.model = self.model.to(self.device)
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank],find_unused_parameters=True)
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank,shuffle=True)
        else:
            self.rank=0
            self.train_sampler = None
            self.model = self.model.to(self.device)

        self.start_epoch = 0
        self.current_step=0
        if os.path.exists(os.path.join(self.OUTPUTDIR,"latest_model.pkl")):
            if self.DISTRIBUTED:
                model = self.model.module
            else:
                model = self.model
            statedict = torch.load(os.path.join(self.OUTPUTDIR,"latest_model.pkl"))
            self.current_step = statedict["current_step"]
            self.start_epoch = self.current_step//self.steps_per_epoch
            model.load_state_dict(statedict["model_state_dict"])
            print("loaded model state from step: ",self.current_step)
            for _ in range(self.current_step):
                self.scheduler.step()
        elif hasattr(self,"WHISPER_PATH"):    
            print("No model checkpoints found,loading whisper checkpoint")
            checkpoint_state_dict = torch.load(self.WHISPER_PATH)

            if self.DISTRIBUTED:
                model = self.model.module
            else:
                model = self.model
            # Load only the matching keys and print the ignored ones
            model_state_dict = model.state_dict()
            matched_keys = []
            ignored_keys = []
            size_mismatch_keys = []

            for name, param in checkpoint_state_dict.items():
                if name in model_state_dict:
                    if param.size() == model_state_dict[name].size():
                        matched_keys.append(name)
                        model_state_dict[name] = param
                    else:
                        size_mismatch_keys.append(name)
                else:
                    ignored_keys.append(name)

            model.load_state_dict(model_state_dict, strict=False)

            # Print the ignored keys and size mismatch keys
            print(f"Ignored keys from the checkpoint:")
            for key in ignored_keys:
                print(key)

            print(f"\nKeys with size mismatch:")
            for key in size_mismatch_keys:
                print(key)
        if self.FREEZE_ENCODER:
            print("freezing encoder layer....")
            for name, params in self.model.named_parameters():
                if "feature_extractor" in name:
                    params.requires_grad = False
                    params.requires_grad_(False)
        collate_func=None
        if hasattr(self,"dataloder_collate"):
            print("Using collate function..")
            collate_func= self.dataloder_collate
        if hasattr(self,"FINETUNING_STRATEGY"):
            self.finetuning_strategy=self.FINETUNING_STRATEGY
        else:
            self.finetuning_strategy = None
        self.train_loader = DataLoader(self.train_dataset,collate_fn=collate_func ,batch_size=self.SAMPLES_PER_GPU, sampler=self.train_sampler, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS)
        
        os.makedirs(self.OUTPUTDIR,exist_ok=True)
        self.logger = setup_logger(os.path.join(self.OUTPUTDIR,"logs.txt"))
        self.metrics = MetricsStore()

        if self.rank==0:
            self.valid_loader = DataLoader(self.valid_dataset,collate_fn=collate_func, batch_size=self.VALIDATION_BS, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS_VAL)
            self.evaluation_callback = ModelValidationCallback(self,self.metrics,self.valid_loader,self.tokenizer,self.PAD_TOKEN)
            if hasattr(self,"ood_dataset"):
                self.evaluation_callback_ood = LongFormatExamplesEvaluation(self,self.metrics,self.ood_dataset,self.tokenizer,self.PAD_TOKEN,window_size=self.OOD_EVALUATION_WINDOW,overlap=self.OOD_EVALUATION_OVERLAP,mel=False)
            else:
                self.evaluation_callback_ood=None
            print("Autoregressive inference:",self.augoregressive_inference)
        if self.AUTOCAST:
            self.train_context = torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
        else:
            self.train_context = nullcontext()
    #@profile
    def train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tqdm_loader = tqdm(self.train_loader,desc=f"Train epoch: {epoch}",disable=self.rank!=0)
        updatefreq=5
        for i,batch in enumerate(tqdm_loader):
            # Your training code here
            self.optimizer.zero_grad()

            with self.train_context:
                inputs, target_tokens ,inplen,targlen= [b.to(self.device) for b in batch]
                outputs = self.model(inputs)
                loss = self.criterion(outputs, target_tokens,inplen,targlen)
            if self.AUTOCAST:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            if i%updatefreq==0:
                if torch.isnan(loss):
                    print("Found nan loss")
                    exit()
                tqdm_loader.set_description(f"loss: {loss.item():.4f} ")
            num_batches += 1
            self.current_step+=1
            if self.current_step%self.VALIDATION_FREQUENCY==0:
                dist.barrier()
                if self.rank == 0:
                    self.validate(self.current_step)
                    avg_loss = total_loss / num_batches
                    self.metrics(self.current_step,"training_loss",avg_loss)
                    self.logger.info(f"###Iter: {self.current_step}  ::  {self.metrics.get_metrics_by_epoch(self.current_step)}")
                    num_batches=0
                    total_loss=0.0
                dist.barrier()

    def validate(self,cs):
        self.model.eval()
        self.evaluation_callback(cs,target_token_index=1)
        self.evaluation_callback_ood(cs)
    def infer(self, inputs):
        return self._inferonepass(inputs)
    
    def _inferonepass(self, inputs):
        if self.DISTRIBUTED:
            model = self.model.module
            
        else:
            model = self.model
        
        all_indices = torch.argmax(model(inputs).detach().cpu(), dim=-1)
        generated = []
        for indices in all_indices:
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = indices[indices != self.BLANK_TOKEN]
            generated.append(indices)
        return generated
        
    def get_state_dict(self):
        if self.DISTRIBUTED:
            model = self.model.module.state_dict()  
        else:
            model = self.model.state_dict()
        return model
    
    def train(self):
        if self.rank==0:
            print("Starting training....")
        
        for epoch in range(self.start_epoch,self.EPOCHS):
            if self.DISTRIBUTED:
                self.train_sampler.set_epoch(epoch)
            if self.FREEZE_ENCODER and  epoch>=self.ENCODER_UNFREEZE_EPOCH:
                print("unfreezing encoder layers....")
                for _, params in self.model.named_parameters():
                    params.requires_grad = True
                    params.requires_grad_(True)
                dist.barrier()
            if self.finetuning_strategy:
                self.unfreeze(epoch)
                dist.barrier()
            self.train_one_epoch(epoch)
        if self.rank==0:
            self.metrics.to_dataframe().to_csv(os.path.join(self.OUTPUTDIR,"metrics.csv"))