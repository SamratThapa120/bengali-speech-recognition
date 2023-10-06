from transformers import Wav2Vec2FeatureExtractor

class AudioPreprocessor:
    def __init__(self,sr=16000):
        self.processor = Wav2Vec2FeatureExtractor()
        self.sr = sr
    def __call__(self, audio):
        return self.processor(audio,sampling_rate=self.sr)["input_values"][0]