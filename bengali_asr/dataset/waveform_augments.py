from typing import Any
import numpy as np
import librosa
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Shift,Resample

# Unused, the quality deteriorates after this transform
        
from transformers import Wav2Vec2Processor

class AudioPreprocessor:
    def __init__(self,pretrained="facebook/wav2vec2-base-100h",sr=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained)
        self.sr = sr
    def __call__(self, audio):
        return self.processor(audio,sr=self.sr)["input_values"][0]

class ConcatTransform:
    def __init__(self, paths,transcripts,max_transcript_length=240,max_concat_len=1,prob=0.5,separator=" "):
        self.paths = paths
        self.transcripts = transcripts
        self.lengths = np.array([len(x) for x in self.transcripts])
        self.max_concat_len=max_concat_len
        self.max_transcript_length = max_transcript_length
        self.separator = separator
        self.prob = prob
    def __call__(self, audio,transcripts):
        if np.random.rand()<self.prob:
            for _ in range(self.max_concat_len):
                tlen = len(transcripts)
                remaining = self.max_transcript_length - tlen
                valid_idxs=np.where(self.lengths<remaining)[0]
                if len(valid_idxs)<1:
                    break
                idx = np.random.choice(valid_idxs)
                taudio,tscript = np.load(self.paths[idx]),self.transcripts[idx]
                transcripts = transcripts + self.separator + tscript
                audio = np.concatenate([audio,taudio],axis=0)
        return audio,transcripts
        
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
        
class AddNaturalNoise:
    def __init__(self, noisefiles, min_portion=0.2, p=0.5, blend_ratio=0.5, blend_count=2):
        self.noise_files = noisefiles
        self.min_portion = min_portion
        self.prob = p
        self.blend_ratio = blend_ratio
        self.blend_count = blend_count

    def __call__(self, data):
        if np.random.rand() < self.prob:
            count = np.random.randint(1,self.blend_count)
            samples = np.random.choice(self.noise_files, count)
            audios = [np.load(f) for f in samples]
            for audio in audios:
                tl = len(data)
                al = len(audio)
                
                # Clip a random part of audio
                if tl<=al:
                    min_len = int(self.min_portion * tl)
                    max_len = tl
                else:
                    min_len = min(int(self.min_portion * tl),al)
                    max_len = al
                clip_len = np.random.randint(min_len, max_len + 1)
                start_idx = np.random.randint(0, al - clip_len + 1)
                clipped_audio = audio[start_idx:start_idx + clip_len]
                
                # Blend to data  at random starting position
                blend_start = np.random.randint(0, tl - clip_len + 1)
                data[blend_start:blend_start + clip_len] = (
                    self.blend_ratio * data[blend_start:blend_start + clip_len] +
                    (1 - self.blend_ratio) * clipped_audio
                )
            return data
        else:
            return data
        
import cv2

class TimeAugment:
    def __init__(self, range=(0.9,1.1),p=0.5,*args,** kwargs):
        self.min = range[0]
        self.range = (range[1]-range[0])
        self.prob = p
    def __call__(self, data):
        if np.random.rand()<self.prob:
            value = self.min+np.random.rand()*self.range
            newlen = int(value*len(data))
            return cv2.resize(data,(1,newlen)).squeeze()
        return data

class PitchShiftAug:
    def __init__(self, range=(1,4),p=0.5,*args,** kwargs):
        self.min = range[0]
        self.max = range[1]
        self.prob = p
    def __call__(self, data):
        if np.random.rand()<self.prob:
            value = np.random.randint()
            return librosa.effects.time_stretch(data,rate=value)
        return data


class PitchShiftAugResampleAugmentation:
    def __init__(self, orig_sr=16000, p=0.5,target_sr_range=(4000,16000)):
        self.orig_sr = orig_sr
        self.min_sr = target_sr_range[0]//1000
        self.max_sr = target_sr_range[1]//1000
        self.prob = p
    def __call__(self, data):
        if np.random.rand()<self.prob:
            target_sr = np.random.randint(self.min_sr,self.max_sr)*1000
            downsampled_data = librosa.resample(data, orig_sr=self.orig_sr, target_sr=target_sr)
            return librosa.resample(downsampled_data, orig_sr=target_sr, target_sr=self.orig_sr)
        else:
            return data