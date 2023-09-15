import sys
import os
import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append("../")

MODEL_ROOT="/app/bengali-speech-recognition/workdir/whisperbase_characterlevel"
MODEL_TYPE="bestmodel_wer.pkl"

df = pd.read_csv("/app/dataset/valid_data.csv")
ROOT_DIR="/app/dataset/train_numpy_16k"

os.environ["CUDA_VISIBLE_DEVICES"]="7"
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.whisper_characterwise import Configs
files = df.id.apply(lambda x: os.path.join(ROOT_DIR,x.replace(".mp3",".npy")))
CFG = Configs(files,df.sentence,True)

data_loader = DataLoader(CFG.inference_dataset,batch_size=64, pin_memory=True, num_workers=8)
device = "cuda"
CFG.load_state_dict(os.path.join(MODEL_ROOT,MODEL_TYPE))
CFG.model.to(device)
1

def infer(inputs):
    batch_size = inputs.size(0)
    generated_tokens = torch.ones((batch_size, 1), dtype=torch.long, device=device) * CFG.START_TOKEN
    encoded_logits = CFG.model.encoder(inputs)
    eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(CFG.MAX_PREDICTION_LENGTH):
        logits = CFG.model.decoder(generated_tokens, encoded_logits)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        # Update end-of-sequence flags
        eos_flags = eos_flags | (next_token.squeeze(-1) == CFG.END_TOKEN)

        # Stop condition: if all sequences in the batch have generated <eos>
        if eos_flags.all():
            break
    return generated_tokens

truths = []
predictions = []

with torch.no_grad():
    for batch in tqdm(data_loader):
        inputs, _, target_tokens = batch

        # Initialize tokens (assuming <sos> token is 0)
        generated_tokens = infer(inputs.to(device)).detach().cpu()

        generated_tokens = generated_tokens[:, 1:]  # Remove the start token
        for gen,tar in zip(generated_tokens,target_tokens):
            end_pos = (gen == CFG.END_TOKEN).nonzero(as_tuple=True)[0]
            if len(end_pos) > 0:
                gen = gen[:end_pos[0]] 
            hypothesis = CFG.tokenizer.decode_torch_inference(gen)
            reference = CFG.tokenizer.decode_torch_inference(tar[tar!=CFG.PAD_TOKEN])

            predictions.append(hypothesis)
            truths.append(reference)
df["predictions"] = predictions
df["truth"] = truths

df.to_csv(os.path.join(MODEL_ROOT,"all_validation_results.csv"),index=False)