import numpy as np
import librosa

# Unused, the quality deteriorates after this transform
class TimeStretch:
    def __init__(self, rate=(0.95,0.5),prob=0.2):
        self.min = rate[0]
        self.range = rate[1]
        self.prob = prob
    def __call__(self, data):
        if np.random.rand()<self.prob:
            rate = self.min+self.range*np.random.rand()
            return librosa.effects.time_stretch(data, rate)
        return data

class PitchShift:
    def __init__(self, sr, n_steps=1.0,prob=0.2):
        self.sr = sr
        self.n_steps = n_steps
        self.prob = prob
    def __call__(self, data):
        if np.random.rand()<self.prob:
            return librosa.effects.pitch_shift(data, self.sr, self.n_steps)
        return data

class AddNoise:
    def __init__(self, prob=0.2,noise_level=0.02):
        self.noise_level = noise_level
        self.prob = prob
    def __call__(self, data):
        if self.prob<np.random.rand():
            noise = np.random.randn(len(data))
            augmented_data = data + self.noise_level * noise
            return augmented_data.astype(type(data[0]))
        return data

class TimeShift:
    def __init__(self, shift_max=0.2,prob=0.2):
        self.shift_max = shift_max
        self.prob = prob
    def __call__(self, data):
        if self.prob<np.random.rand():
            shift = int(len(data) * self.shift_max * (2 * np.random.rand() - 1))
            return np.roll(data, shift)
        else:
            return data

class ResampleAugmentation:
    def __init__(self, orig_sr=16000, prob=0.2,target_sr_range=(4000,16000)):
        self.orig_sr = orig_sr
        self.min_sr = target_sr_range[0]
        self.max_sr = target_sr_range[1]
        self.prob = prob
    def __call__(self, data):
        if self.prob<np.random.rand():
            target_sr = np.random.randint(self.min_sr,self.max_sr)
            downsampled_data = librosa.resample(data, self.orig_sr, target_sr)
            return librosa.resample(downsampled_data, target_sr, self.orig_sr)
        else:
            return data