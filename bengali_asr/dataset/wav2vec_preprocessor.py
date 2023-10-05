from transformers import Wav2Vec2Processor

class AudioPreprocessor:
    def __init__(self,pretrained="facebook/wav2vec2-base-100h",sr=16000):
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained)
        self.sr = sr
    def __call__(self, audio):
        return self.processor(audio,sr=self.sr)["input_values"][0]