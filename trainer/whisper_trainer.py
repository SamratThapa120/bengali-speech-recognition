import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from .utils import setup_logger,MetricsStore
import os
from tqdm import tqdm
from bengali_asr.callbacks.evaluation import WhisperAutoregressiveEvaluation

class Trainer:
    def __init__(self, base_obj):
        self.__dict__.update(base_obj.get_all_attributes())

        if self.DISTRIBUTED:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"
            self.model = self.model.to(self.device)
            self.model = DistributedDataParallel(self.model, device_ids=[self.rank],)
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank)
        else:
            self.rank=0
            self.train_sampler = None
            self.model = self.model.to(self.device)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.SAMPLES_PER_GPU, sampler=self.train_sampler, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS)
        
        os.makedirs(self.OUTPUTDIR,exist_ok=True)
        self.logger = setup_logger(os.path.join(self.OUTPUTDIR,"logs.txt"))
        self.metrics = MetricsStore()

        if self.rank==0:
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.VALIDATION_BS, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS)
            self.evaluation_callback = WhisperAutoregressiveEvaluation(self,self.metrics,self.valid_loader,self.tokenizer,self.PAD_TOKEN)
       

    #@profile
    def train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        tqdm_loader = tqdm(self.train_loader,desc=f"Train epoch: {epoch}",disable=self.rank!=0)
        updatefreq=5
        for i,batch in enumerate(tqdm_loader):
            # Your training code here
            inputs, inp_tokens,target_tokens = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs,inp_tokens)
            loss = self.criterion(outputs, target_tokens)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            if i%updatefreq==0:
                tqdm_loader.set_description(f"loss: {loss.item():.4f} ")
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.metrics(epoch,"training_loss",avg_loss)

    def validate(self,epoch):
        if self.rank != 0 or epoch%self.VALIDATION_FREQUENCY!=0:
            return
        self.model.eval()
        self.evaluation_callback(epoch)

    def infer(self, inputs):
        if self.DISTRIBUTED:
            model = self.model.module
            
        else:
            model = self.model
        
        batch_size = inputs.size(0)
        generated_tokens = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * self.START_TOKEN
        encoded_logits = model.encoder(inputs)
        eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(self.MAX_PREDICTION_LENGTH):
            logits = model.decoder(generated_tokens, encoded_logits)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

            # Update end-of-sequence flags
            eos_flags = eos_flags | (next_token.squeeze(-1) == self.END_TOKEN)

            # Stop condition: if all sequences in the batch have generated <eos>
            if eos_flags.all():
                break
        return generated_tokens

    def get_state_dict(self):
        if self.DISTRIBUTED:
            model = self.model.module.state_dict()  
        else:
            model = self.model.state_dict()
        return model
    
    def train(self):
        if self.rank==0:
            print("Starting training....")
        for epoch in range(self.EPOCHS):
            if self.DISTRIBUTED:
                self.train_sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            self.validate(epoch)
            if self.rank==0:
                self.logger.info(f"###Epoch: {epoch}  ::  {self.metrics.get_metrics_by_epoch(epoch)}")
        if self.rank==0:
            self.metrics.to_dataframe().to_csv(os.path.join(self.OUTPUTDIR,"metrics.csv"))