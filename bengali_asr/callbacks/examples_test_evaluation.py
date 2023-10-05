

from jiwer import wer, cer
import torch
from tqdm import tqdm
import os
import numpy as np

def sliding_window(tensor, window_size, overlap):
    """
    Slices the input tensor with a sliding window of given size and overlap.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape (points,).
        window_size (int): The size of the sliding window.
        overlap (int): The overlap between consecutive windows.
        
    Returns:
        torch.Tensor: A tensor containing the sliced windows stacked along the 0 dimension.
    """
    step = window_size - overlap
    slices = []
    
    for start in range(0, len(tensor) - window_size + 1, step):
        end = start + window_size
        slices.append(tensor[start:end])
    
    # Handle the final feature and pad it with zeros to match window_size
    if len(tensor) % step != 0:
        last_slice = tensor[-window_size:]
        slices.append(torch.nn.functional.pad(last_slice, (0, window_size - len(last_slice))))
    
    return torch.stack(slices, dim=0)

def sliding_window_spectro(tensor, window_size, overlap):
    """
    Slices the input tensor with a sliding window of given size and overlap.
    
    Args:
        tensor (torch.Tensor): The input tensor of shape (mels,points).
        window_size (int): The size of the sliding window.
        overlap (int): The overlap between consecutive windows.
        
    Returns:
        torch.Tensor: A tensor containing the sliced windows stacked along the 0 dimension.
    """
    step = window_size - overlap
    slices = []
    
    for start in range(0, tensor.shape[1] - window_size + 1, step):
        end = start + window_size
        slices.append(tensor[:,start:end])
    
    # Handle the final feature and pad it with zeros to match window_size
    if tensor.shape[1] % step != 0:
        last_slice = tensor[:,-window_size:]
        slices.append(torch.nn.functional.pad(last_slice, (0, window_size - last_slice.shape[1])))
    
    return torch.stack(slices, dim=0)

class LongFormatExamplesEvaluation:
    def __init__(self,model,metrics,valid_dset,tokenizer,ignore_token,save_ratio=1,window_size=1700,overlap=1,save_cer=False,mel=True):
        self.model = model
        self.metrics = metrics
        self.valid_dset = valid_dset
        self.tokenizer = tokenizer
        self.ignore_token = ignore_token
        self.current_best_wer = 100000
        self.current_best_cer= 100000
        self.save_ratio = save_ratio
        self.window_size = window_size
        self.overlap = overlap
        self.ground_truths = self.valid_dset.transcripts
        self.save_cer = save_cer
        self.mel = mel
    def _savemodel(self,current_step,path):
        torch.save({
            'current_step': current_step,
            'model_state_dict': self.model.get_state_dict(),
        }, path)
    #@profile
    def __call__(self, current_step):
        total_wer = 0
        total_cer = 0
        total_samples = 0

        predictions = []

        with torch.no_grad():
            for batch,truth in tqdm(zip(self.valid_dset,self.ground_truths),desc=f"Valid step: {current_step}"):
                if self.mel:
                    inputs = sliding_window_spectro(batch[0],self.window_size,self.overlap)
                else:
                    inputs = sliding_window(batch[0],self.window_size,self.overlap)
                # Initialize tokens (assuming <sos> token is 0)
                generated_tokens = self.model.infer(inputs.to(self.model.device))
                generated = []
                for gen in generated_tokens:
                    hypothesis = self.tokenizer.decode_torch_inference(gen[1:])
                    generated.append(hypothesis)
                preds = "".join(generated)
                predictions.append(preds)
                total_wer+= wer(truth,preds)
                total_cer+= cer(truth,preds)
                total_samples+=1
        avg_wer = total_wer / total_samples
        avg_cer = total_cer / total_samples
        self.metrics(current_step,"word_error_rate_ood",avg_wer)
        self.metrics(current_step,"char_error_rate_ood",avg_cer)

        if avg_wer<=self.current_best_wer:
            print("saving best wer model")
            with open(os.path.join(self.model.OUTPUTDIR,"predictions_wer_ood.txt"),"w") as f:
                for t,p in zip(self.ground_truths,predictions):
                    f.write(f"===========================================================================================================================\n")
                    f.write(f"Truth: {t}\n")
                    f.write(f"---------------------------------------------------------------------------------------------------------------------------\n")
                    f.write(f"Prediction: {p}\n")
            self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"bestmodel_wer_ood.pkl"))
            self.current_best_wer = avg_wer

        if avg_cer<=self.current_best_cer and self.save_cer:
            print("saving best cer model")
            with open(os.path.join(self.model.OUTPUTDIR,"predictions_cer_ood.txt"),"w") as f:
                for t,p in zip(self.ground_truths,predictions):
                    f.write(f"===========================================================================================================================\n")
                    f.write(f"Truth: {t}\n")
                    f.write(f"---------------------------------------------------------------------------------------------------------------------------\n")
                    f.write(f"Prediction: {p}\n")
            self._savemodel(current_step,os.path.join(self.model.OUTPUTDIR,"bestmodel_cer_ood.pkl"))
            self.current_best_cer = avg_cer
            
