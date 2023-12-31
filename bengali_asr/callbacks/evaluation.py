

from jiwer import wer, cer
import torch
from tqdm import tqdm
import os
import numpy as np

class ModelValidationCallback:
    def __init__(self,model,metrics,valid_loader,tokenizer,ignore_token,save_ratio=0.2,save_cer=False):
        self.model = model
        self.metrics = metrics
        self.valid_loader = valid_loader
        self.tokenizer = tokenizer
        self.ignore_token = ignore_token
        self.current_best_wer = 100000
        self.current_best_cer= 100000
        self.save_ratio = save_ratio
        self.save_cer = save_cer
    def _savemodel(self,current_step,path):
        torch.save({
            'current_step': current_step,
            'model_state_dict': self.model.get_state_dict(),
        }, path)
    #@profile
    def __call__(self, current_step,target_token_index=-1):
        self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"latest_model.pkl"))
        total_wer = 0
        total_cer = 0
        total_samples = 0

        truths = []
        predictions = []

        with torch.no_grad():
            for batch in tqdm(self.valid_loader,desc=f"Valid step: {current_step}"):
                inputs= batch[0]
                target_tokens = batch[target_token_index] 
                # Initialize tokens (assuming <sos> token is 0)
                generated_tokens = self.model.infer(inputs.to(self.model.device))
                for gen,tar in zip(generated_tokens,target_tokens):
                    hypothesis = self.tokenizer.decode_torch_inference(gen)
                    reference = self.tokenizer.decode_torch_inference(tar[tar!=self.ignore_token])
                    if np.random.rand()<self.save_ratio:
                        predictions.append(hypothesis)
                        truths.append(reference)

                    total_wer += wer(reference, hypothesis)
                    total_cer += cer(reference, hypothesis)
                    total_samples += 1

        avg_wer = total_wer / total_samples
        avg_cer = total_cer / total_samples
        self.metrics(current_step,"word_error_rate",avg_wer)
        self.metrics(current_step,"char_error_rate",avg_cer)

        if avg_wer<=self.current_best_wer:
            print("saving best wer model")
            with open(os.path.join(self.model.OUTPUTDIR,"predictions_wer.txt"),"w") as f:
                for t,p in zip(truths,predictions):
                    f.write(f"===========================================================================================================================\n")
                    f.write(f"Truth: {t}\n")
                    f.write(f"---------------------------------------------------------------------------------------------------------------------------\n")
                    f.write(f"Prediction: {p}\n")
            self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"bestmodel_wer.pkl"))
            self.current_best_wer = avg_wer

        if avg_cer<=self.current_best_cer and self.save_cer:
            print("saving best cer model")
            with open(os.path.join(self.model.OUTPUTDIR,"predictions_cer.txt"),"w") as f:
                for t,p in zip(truths,predictions):
                    f.write(f"===========================================================================================================================\n")
                    f.write(f"Truth: {t}\n")
                    f.write(f"---------------------------------------------------------------------------------------------------------------------------\n")
                    f.write(f"Prediction: {p}\n")
            self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"bestmodel_cer.pkl"))
            self.current_best_cer = avg_cer
            
