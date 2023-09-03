

from jiwer import wer, cer
import torch
from tqdm import tqdm
import os

class WhisperAutoregressiveEvaluation:
    def __init__(self,model,metrics,valid_loader,tokenizer,ignore_token):
        self.model = model
        self.metrics = metrics
        self.valid_loader = valid_loader
        self.tokenizer = tokenizer
        self.ignore_token = ignore_token
        self.current_best_wer = 100000
        self.current_best_cer= 100000
    
    #@profile
    def __call__(self, epoch):
        total_wer = 0
        total_cer = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_loader,desc=f"Valid epoch: {epoch}"):
                inputs, _, target_tokens = batch

                # Initialize tokens (assuming <sos> token is 0)
                generated_tokens = self.model.infer(inputs.to(self.model.device)).detach().cpu()

                generated_tokens = generated_tokens[:, 1:]  # Remove the start token
                for gen,tar in zip(generated_tokens,target_tokens):
                    end_pos = (gen == self.model.END_TOKEN).nonzero(as_tuple=True)[0]
                    if len(end_pos) > 0:
                        gen = gen[end_pos[0]:] 
                    hypothesis = self.tokenizer.decode_torch_inference(gen)
                    reference = self.tokenizer.decode_torch_inference(tar[tar!=self.ignore_token])

                    total_wer += wer(reference, hypothesis)
                    total_cer += cer(reference, hypothesis)
                    total_samples += 1

        avg_wer = total_wer / total_samples
        avg_cer = total_cer / total_samples
        self.metrics(epoch,"word_error_rate",avg_wer)
        self.metrics(epoch,"char_error_rate",avg_cer)

        if avg_wer<=self.current_best_wer:
            print("saving best wer model")
            torch.save(self.model.model.state_dict(),os.path.join(self.model.OUTPUTDIR,"bestmodel_wer.pkl"))
            self.current_best_wer = avg_wer
        if avg_cer<=self.current_best_cer:
            print("saving best cer model")
            torch.save(self.model.model.state_dict(),os.path.join(self.model.OUTPUTDIR,"bestmodel_cer.pkl"))
            self.current_best_cer = avg_cer