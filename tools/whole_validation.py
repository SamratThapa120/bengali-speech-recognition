import sys
import os
import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("../")

MODEL_ROOT="/app/bengali-speech-recognition/workdir/whisperbase_characterlevel_finetuned_nolm_posembd_ctcloss"
MODEL_TYPE="bestmodel_wer.pkl"

df = pd.read_csv("/app/dataset/valid_data.csv")
ROOT_DIR="/app/dataset/train_numpy_16k"

os.environ["CUDA_VISIBLE_DEVICES"]="7"
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.whisper_characterwise_nolm_ctcloss import Configs
files = df.id.apply(lambda x: os.path.join(ROOT_DIR,x.replace(".mp3",".npy")))
CFG = Configs(files,df.sentence,True)

data_loader = DataLoader(CFG.inference_dataset,batch_size=64, pin_memory=True, num_workers=8)
device = "cuda"
CFG.load_state_dict(os.path.join(MODEL_ROOT,MODEL_TYPE))
CFG.model.to(device)
1

def infer(inputs):
    all_indices = torch.argmax(CFG.model(inputs).detach().cpu(), dim=-1)
    generated = []
    for indices in all_indices:
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = indices[indices != CFG.BLANK_TOKEN]
        generated.append(indices)
    return generated

truths = []
predictions = []

with torch.no_grad():
    for batch in tqdm(data_loader):
        inputs = batch[0]
        target_tokens = batch[-1]

        # Initialize tokens (assuming <sos> token is 0)
        generated_tokens = infer(inputs.to(device))
        for gen,tar in zip(generated_tokens,target_tokens):

            hypothesis = CFG.tokenizer.decode_torch_inference(gen)
            reference = CFG.tokenizer.decode_torch_inference(tar[tar!=CFG.PAD_TOKEN])

            predictions.append(hypothesis)
            truths.append(reference)
df["predictions"] = predictions
df["truth"] = truths

df.to_csv(os.path.join(MODEL_ROOT,"all_validation_results.csv"),index=False)