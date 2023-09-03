import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from jiwer import wer, cer

class Trainer:
    def __init__(self, base_obj):
        self.__dict__.update(base_obj.get_all_attributes())
        
        if self.DISTRIBUTED:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank)
        else:
            self.rank=0
            self.train_sampler = None

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.SAMPLES_PER_GPU, sampler=self.train_sampler, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS)
        if self.rank==0:
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.VALIDATION_BS, pin_memory=self.PIN_MEMORY, num_workers=self.NUM_WORKERS)

    def train_one_epoch(self,epoch):
        self.model.train()
        for batch in self.train_loader:
            # Your training code here
            inputs, inp_tokens,target_tokens = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs,inp_tokens)
            loss = self.criterion(outputs, target_tokens)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def validate(self,epoch):
        if self.rank != 0 and self.epoch%self.VALIDATION_FREQUENCY==0:
            return

        self.model.eval()
        total_wer = 0
        total_cer = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                inputs, _, target_tokens = [b.to(self.device) for b in batch]
                batch_size = inputs.size(0)
                assert batch_size == 1, "Batch size for validation must be 1 for autoregressive inference"

                # Initialize tokens (assuming <sos> token is 0)
                generated_tokens = torch.ones((batch_size, 1), dtype=torch.long, device=self.device)*self.START_TOKEN
                encoded_logits = self.model.encoder(inputs)
                for step in range(self.MAX_PREDICTION_LENGTH):  
                    logits = self.model.decoder(generated_tokens,encoded_logits)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

                    # Stop condition (assuming <eos> token is 1)
                    if next_token.item() == self.END_TOKEN:
                        break

                # Calculate WER and CER
                hypothesis = self.tokenizer.decode_torch_inference(generated_tokens[0])
                reference = self.tokenizer.decode_torch_inference(target_tokens[0])
                total_wer += wer(reference, hypothesis)
                total_cer += cer(reference, hypothesis)
                total_samples += 1

        avg_wer = total_wer / total_samples
        avg_cer = total_cer / total_samples

        print(f"Epoch: {epoch}, Validation WER: {avg_wer}, Validation CER: {avg_cer}")

    def train(self):
        print("Starting training....")
        for epoch in range(self.EPOCHS):
            if self.DISTRIBUTED:
                self.train_sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            self.validate(epoch)