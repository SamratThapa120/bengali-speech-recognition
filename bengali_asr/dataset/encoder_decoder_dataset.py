import os
import torch
from torch.utils.data import Dataset
from subprocess import CalledProcessError, run
import numpy as np
from torch.nn import functional as F

#@profile
def load_audio(file: str, sr: int):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

class SpeechRecognitionDataset(Dataset):
    def __init__(self,files,transcript,tokenizer,root="",
                 raw_transform=None,mel_transform=None,sampling_rate=16000,
                 token_length=256,pad_token=-1,train=True,usenumpy=True):
        """
        Args:
            root_dir (str): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.speech_files = [os.path.join(root,f) for f in files]
        self.transcripts = transcript
        self.raw_transform = raw_transform
        self.mel_transform = mel_transform
        self.tokenizer = tokenizer
        self.sr = sampling_rate
        self.token_length = token_length
        self.pad_token = pad_token
        self.train=train
        self.usenumpy = usenumpy
        
    def __len__(self):
        return len(self.speech_files)
    
    #@profile
    def __getitem__(self, idx):
        if self.usenumpy:
            audio_raw = np.load(self.speech_files[idx])
        else:
            audio_raw = load_audio(self.speech_files[idx],self.sr)

        if self.raw_transform:
            audio_raw = self.raw_transform(audio_raw)
            
        tensor = self.mel_transform(audio_raw)
        tokens = self.tokenizer(self.transcripts[idx],self.train)
        if not self.train:
            pad_len_output = self.token_length - len(tokens)
            if pad_len_output > 0:
                tokens = F.pad(tokens, (0, pad_len_output), value=self.pad_token)
            else:
                tokens = tokens[:self.token_length]
            return torch.from_numpy(tensor),torch.tensor([]),tokens
        
        input_tokens = tokens[:-1]
        output_tokens = tokens[1:]
        
        # Pad input_tokens and output_tokens if they are shorter than self.token_length
        pad_len_input = self.token_length - len(input_tokens)
        pad_len_output = self.token_length - len(output_tokens)
        if pad_len_input > 0:
            input_tokens = F.pad(input_tokens, (0, pad_len_input), value=self.tokenizer.end_token)
        else:
            input_tokens = input_tokens[:self.token_length]

        if pad_len_output > 0:
            output_tokens = F.pad(output_tokens, (0, pad_len_output), value=self.pad_token)
        else:
            output_tokens = output_tokens[:self.token_length]

        return torch.from_numpy(tensor),input_tokens,output_tokens

class SpeechRecognitionDatasetSimplified(Dataset):
    def __init__(self,files,transcript,tokenizer,root="",
                 raw_transform=None,mel_transform=None,concat_transform=None,sampling_rate=16000,
                 token_length=256,pad_token=-1,train=True,usenumpy=True):
        """
        Args:
            root_dir (str): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.speech_files = [os.path.join(root,f) for f in files]
        self.transcripts = transcript
        self.raw_transform = raw_transform
        self.mel_transform = mel_transform
        self.tokenizer = tokenizer
        self.sr = sampling_rate
        self.token_length = token_length
        self.pad_token = pad_token
        self.train=train
        self.usenumpy = usenumpy
        self.concat_transform=concat_transform
    def __len__(self):
        return len(self.speech_files)
    
    #@profile
    def __getitem__(self, idx):
        transcript = self.transcripts[idx]
        if self.usenumpy:
            audio_raw = np.load(self.speech_files[idx])
        else:
            audio_raw = load_audio(self.speech_files[idx],self.sr)

        if self.raw_transform:
            audio_raw = self.raw_transform(audio_raw)
        if self.concat_transform:
            audio_raw,transcript = self.concat_transform(audio_raw,transcript) 
        tensor = self.mel_transform(audio_raw)
        tokens = self.tokenizer(transcript,self.train)
        return torch.from_numpy(tensor),tokens



class SpeechRecognitionCTCDataset(SpeechRecognitionDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __getitem__(self, idx):
        if self.usenumpy:
            audio_raw = np.load(self.speech_files[idx])
        else:
            audio_raw = load_audio(self.speech_files[idx],self.sr)

        if self.raw_transform:
            audio_raw = self.raw_transform(audio_raw)
            
        tensor = self.mel_transform(audio_raw)
        tokens = self.tokenizer(self.transcripts[idx],self.train)
        
        pad_len_output = self.token_length - len(tokens)
        if pad_len_output > 0:
            tokens = F.pad(tokens, (0, pad_len_output), value=self.pad_token)
        else:
            tokens = tokens[:self.token_length]
        return torch.from_numpy(tensor),tokens
    
class SpeechRecognitionWhisperCollate:
    def __init__(self,max_token_length=256,
                  max_ctx_length=3000,
                  n_mels=80,
                  audio_padding=0.0,
                  token_padding=-1,
                  inp_token_padding=-1
                  ):
        self.max_token_length = max_token_length
        self.max_ctx_length = max_ctx_length
        self.audio_padding = audio_padding
        self.token_padding = token_padding
        self.inp_token_padding = inp_token_padding
        self.n_mels = n_mels
    def __call__(self, batch):
        # Separate the audio and tokens
        audios, tokens_list = zip(*batch)

        # Find the max lengths in the batch
        # max_audio_len = self.max_audio_length
        max_audio_len = min(self.max_ctx_length, max([audio.shape[1] for audio in audios]))
        max_token_len = min(self.max_token_length, max([len(tokens) for tokens in tokens_list]))

        # Initialize tensors for padded audios and tokens
        audios_padded = torch.full((len(audios), self.n_mels ,max_audio_len), self.audio_padding)
        inp_tokens_padded = torch.full((len(tokens_list), max_token_len), self.inp_token_padding)

        tokens_padded = torch.full((len(tokens_list), max_token_len), self.token_padding)
        # Pad the audios and tokens
        for idx, (audio, tokens) in enumerate(zip(audios, tokens_list)):
            audio_len = min(audio.shape[1], max_audio_len)
            token_len = min(len(tokens), max_token_len)

            audios_padded[idx, :,:audio_len] = audio[:,:audio_len]
            inp_tokens_padded[idx, :token_len] = tokens[:token_len]
            tokens_padded[idx, :token_len] = tokens[:token_len]

        return audios_padded, inp_tokens_padded[:,:-1],tokens_padded[:,1:]