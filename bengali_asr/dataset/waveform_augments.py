from typing import Any
import numpy as np
import librosa
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Shift,Resample

# Unused, the quality deteriorates after this transform
        
class GaussianNoise:
    def __init__(self, sr=16000,p=0.5,*args,** kwargs):
        self.sr = sr
        self.aug = AddGaussianNoise(p=1,*args,** kwargs)
        self.prob = p
    def __call__(self, data):
        if np.random.rand()<self.prob:
            self.aug.randomize_parameters(data,self.sr)
            return self.aug.apply(data,self.sr)
        else:
            return data
    
class TimeStretchAug:
    def __init__(self, sr=16000,leave_length_unchanged=False,*args,** kwargs):
        self.sr = sr
        self.aug = TimeStretch(leave_length_unchanged=leave_length_unchanged,*args,** kwargs)
    def __call__(self, data):
        self.aug.randomize_parameters(data,self.sr)
        return self.aug.apply(data,self.sr)

class PitchShiftAug:
    def __init__(self, sr=16000,*args,** kwargs):
        self.sr = sr
        self.aug = PitchShift(*args,** kwargs)
    def __call__(self, data):
        self.aug.randomize_parameters(data,self.sr)
        return self.aug.apply(data,self.sr)


class ResampleAugmentation:
    def __init__(self, orig_sr=16000, p=0.5,target_sr_range=(4000,16000)):
        self.orig_sr = orig_sr
        self.min_sr = target_sr_range[0]//1000
        self.max_sr = target_sr_range[1]//1000
        self.prob = p
    def __call__(self, data):
        if np.random.rand()<self.prob:
            target_sr = np.random.randint(self.min_sr,self.max_sr)*1000
            print(target_sr)
            downsampled_data = librosa.resample(data, orig_sr=self.orig_sr, target_sr=target_sr)
            return librosa.resample(downsampled_data, orig_sr=target_sr, target_sr=self.orig_sr)
        else:
            return data