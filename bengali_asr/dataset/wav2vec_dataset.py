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
import torch

class SpeechRecognitionCollate:
    def __init__(self, max_token_length=256, max_audio_length=400000, audio_padding=0.0, token_padding=-1,audio_scale=320):
        self.max_token_length = max_token_length
        self.max_audio_length = max_audio_length
        self.audio_padding = audio_padding
        self.token_padding = token_padding
        self.audio_scale=audio_scale
        self.max_audio_encoded_length = (max_audio_length//self.audio_scale)-1
    def __call__(self, batch):
        # Separate the audio and tokens
        audios, tokens_list = zip(*batch)

        # Find the max lengths in the batch
        # max_audio_len = self.max_audio_length
        max_audio_len = min(self.max_audio_length, max([len(audio) for audio in audios]))
        max_token_len = min(self.max_token_length, max([len(tokens) for tokens in tokens_list]))

        # Initialize tensors for padded audios and tokens
        audios_padded = torch.full((len(audios), max_audio_len), self.audio_padding)
        tokens_padded = torch.full((len(tokens_list), max_token_len), self.token_padding)

        audio_lengths = []
        token_lengths = []

        # Pad the audios and tokens
        for idx, (audio, tokens) in enumerate(zip(audios, tokens_list)):
            audio_len = min(len(audio), max_audio_len)
            token_len = min(len(tokens), max_token_len)

            audios_padded[idx, :audio_len] = audio[:audio_len]
            tokens_padded[idx, :token_len] = tokens[:token_len]

            audio_lengths.append(min(audio_len//self.audio_scale,self.max_audio_encoded_length))
            token_lengths.append(token_len)

        return audios_padded, tokens_padded, torch.tensor(audio_lengths), torch.tensor(token_lengths)

    # Usage:
    # collate_fn = SpeechRecognitionCollate(max_token_length=..., max_audio_length=..., audio_padding=..., token_padding=...)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=..., collate_fn=collate_fn)

class SpeechRecognitionCTCDataset():
    def __init__(self,files,transcript,tokenizer,root="",
                    raw_transform=None,sampling_rate=16000,train=True,usenumpy=True):
            """
            Args:
                root_dir (str): Directory with all the audio files.
                transform (callable, optional): Optional transform to be applied on a sample.
            """
            self.speech_files = [os.path.join(root,f) for f in files]
            self.transcripts = transcript
            self.raw_transform = raw_transform
            self.tokenizer = tokenizer
            self.sr = sampling_rate
            self.train=train
            self.usenumpy = usenumpy
        
    def __len__(self):
        return len(self.speech_files)

    def __getitem__(self, idx):
        if self.usenumpy:
            audio_raw = np.load(self.speech_files[idx])
        else:
            audio_raw = load_audio(self.speech_files[idx],self.sr)

        if self.raw_transform:
            audio_raw = self.raw_transform(audio_raw)
        
        audio_raw = torch.tensor(audio_raw)
        tokens = self.tokenizer(self.transcripts[idx],self.train)
        return audio_raw,tokens