import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import istft, stft
import librosa

def pad_or_trim(array, length: int , *, axis: int = -1):
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


class LogMelSpectrogramTransform:
    def __init__(
        self,
        n_mels: int,
        n_fft: int,
        hoplen: int,
        sampling_rate,
        tensor_length: int = None,
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hoplen = hoplen
        self.sampling_rate = sampling_rate
        self.tensor_length = tensor_length
    
    @lru_cache(maxsize=None)
    def mel_filters(self,n_mels: int) -> np.ndarray:
        # Generate Mel filters using librosa
        mel_filter_bank = librosa.filters.mel(sr=self.sampling_rate, n_fft=self.n_fft, n_mels=n_mels)
        return mel_filter_bank
    
    def __call__(self, audio,axis: int = -1):
        _, _, Zxx = stft(audio, fs=self.sampling_rate, nperseg=self.n_fft, noverlap=self.hoplen, window='hann', return_onesided=True)
        magnitudes = np.abs(Zxx) ** 2

        filters = self.mel_filters(n_mels=self.n_mels)  # Replace None with the appropriate device if needed
        mel_spec = np.dot(filters, magnitudes)

        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
        array = (log_spec + 4.0) / 4.0

        if self.tensor_length:
            if array.shape[axis] > self.tensor_length:
                array = array.take(indices=range(self.tensor_length), axis=axis)

            if array.shape[axis] < self.tensor_length:
                pad_widths = [(0, 0)] * array.ndim
                pad_widths[axis] = (0, self.tensor_length - array.shape[axis])
                array = np.pad(array, pad_widths)
        return array
    
    def inverse(self, log_mel_spec):
        # Step 1: Unscale and unclip
        array = log_mel_spec * 4.0 - 4.0

        # Step 2: Inverse log
        mel_spec = np.power(10, array)

        # Step 3: Inverse Mel filter bank
        filters = self.mel_filters(n_mels=self.n_mels)  # Replace None with the appropriate device if needed
        filters_inv = np.linalg.pinv(filters)
        magnitudes = np.dot(filters_inv, mel_spec)
        # Step 4: Inverse STFT
        # Clip negative values before taking the square root
        magnitudes = np.clip(magnitudes, a_min=0, a_max=None)

        # Check for negative values
        if np.any(magnitudes < 0):
            print("Warning: Negative values found in magnitudes. Clipping to zero.")

        _, audio_reconstructed = istft(np.sqrt(magnitudes), fs=self.sampling_rate, nperseg=self.n_fft, noverlap=self.hoplen, window='hann')

        return audio_reconstructed

