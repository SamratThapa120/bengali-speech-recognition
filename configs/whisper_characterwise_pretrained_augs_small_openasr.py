from bengali_asr.models import ModelDimensions
from bengali_asr.models import Whisper
from bengali_asr.dataset.tokenizer import CharacterLevelTokenizer
from bengali_asr.dataset.encoder_decoder_dataset import SpeechRecognitionDatasetSimplified,SpeechRecognitionWhisperCollate

from bengali_asr.audio import LogMelSpectrogramTransform,PadTruncateSpectrogram
from bengali_asr.dataset.transforms import ComposeAll
from bengali_asr.models.loss import MaskedCrossEntropyLoss
from bengali_asr.dataset.mel_augments import FrequencyMasking,TimeMasking
from bengali_asr.dataset.waveform_augments import GaussianNoise,TimeAugment,PitchShiftAug,ResampleAugmentation
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .base import Base
import os 
class Configs(Base):
    OUTPUTDIR="../workdir/whispersmall_characterlevel_finetuned_augmentations_openasr"
    WHISPER_PATH="/app/bengali-speech-recognition/workdir/whisper_checkpoints/small.pkl"
    TRAIN_DATA_PATH="/app/dataset/train_data_with_openasr.csv"
    VALID_DATA_PATH="/app/dataset/valid_data_subset.csv"
    DATA_ROOT="/app/dataset/train_numpy_16k"
    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=12
    GRADIENT_STEPS=2
    N_GPU=4
    VALIDATION_BS=12
    VALIDATION_FREQUENCY=5000
    
    PIN_MEMORY=True
    NUM_WORKERS=4
    NUM_WORKERS_VAL=4
    DISTRIBUTED=True
    FREEZE_ENCODER=False
    TRAIN_TYPE=""
    LR=0.000025
    WD=1e-5

    EPOCHS=10
    augoregressive_inference=True

    VOCAB = ['ব', 'া', 'ং', 'ল', 'দ', 'ে', 'শ', ' ', 'য', '়', 'ি', 'ত', '্', 'ন', 'এ', 'ধ', 'র', 'ণ', 'ক', 'ড', 'হ', 'উ', 'প', 'জ', 'অ', 'থ', 'স', 'ষ', 'ই', 'আ', 'ছ', 'গ', 'ু', 'ো', 'ও', 'ভ', 'ী', 'ট', 'ূ', 'ম', 'ৈ', 'ৃ', 'ঙ', 'খ', 'ঃ', '১', '৯', '৬', '০', '২', 'চ', 'ঘ', 'ৎ', '৫', '৪', '-', '‘', '’', 'ফ', ',', 'ৌ', '৮', 'ঁ', 'য়', '৩', 'ঢ', 'ঠ', '৭', ':', '।', '.', 'ড়', 'ঝ', '/', 'ঞ', '"', "'", 'ঔ', 'ঈ', 'ঐ','!', 'ঋ', 'ঊ', '?', '–', ';', 'ঢ়', '—']
    
    START_TOKEN=len(VOCAB)
    END_TOKEN=len(VOCAB)+1
    MAX_PREDICTION_LENGTH=256
    MAX_TOKEN_LENGTH=MAX_PREDICTION_LENGTH
    PAD_TOKEN=-1
    AUDIO_PADDING=0.0
    AUTOCAST=False
    def __init__(self,inference_files=None,inference_text=None):
        self.device = "cuda"
        self.dataloder_collate = SpeechRecognitionWhisperCollate(self.MAX_TOKEN_LENGTH,
                                    self.N_FRAMES,
                                    self.N_MELS,
                                    self.AUDIO_PADDING,
                                    self.PAD_TOKEN,
                                    self.END_TOKEN)
        self.model_dims = ModelDimensions(n_mels=self.N_MELS, 
                                    n_audio_ctx=self.N_FRAMES//2, 
                                    n_audio_state=768,
                                    n_audio_head=12,
                                    n_audio_layer=12,
                                    n_vocab=len(self.VOCAB)+2, 
                                    n_text_ctx=448, 
                                    n_text_state=768, 
                                    n_text_head=12, 
                                    n_text_layer=12)
        self.model = Whisper(self.model_dims)
        self.tokenizer = CharacterLevelTokenizer(self.VOCAB,self.START_TOKEN,self.END_TOKEN)
        self.mel_transorm_valid = ComposeAll([
            LogMelSpectrogramTransform(self.N_MELS,self.N_FFT,self.HOP_LENGTH,self.SAMPLE_RATE),
        ])
        if inference_files is not None:
            print("inference mode is on")
            self.inference_dataset = SpeechRecognitionDatasetSimplified(inference_files,
                                        inference_text,
                                        self.tokenizer,
                                        self.DATA_ROOT,mel_transform=self.mel_transorm_valid,
                                        sampling_rate=self.SAMPLE_RATE,token_length=self.MAX_PREDICTION_LENGTH, pad_token=self.PAD_TOKEN,train=False,usenumpy=False) 
            return
        self.audio_transform_train = ComposeAll([
            GaussianNoise(p=0.2)
        ])
        self.mel_transorm_train = ComposeAll([
            LogMelSpectrogramTransform(self.N_MELS,self.N_FFT,self.HOP_LENGTH,self.SAMPLE_RATE),
            FrequencyMasking(prob=0.5),
            TimeMasking(prob=0.5),
        ])
        
        self.training_data = pd.read_csv(self.TRAIN_DATA_PATH)[:self.USE_DATASET_LEN]
        self.valid_data = pd.read_csv(self.VALID_DATA_PATH)[:self.USE_DATASET_LEN]
        print(f"length of train: {len(self.training_data)}, length of valid: {len(self.valid_data)}")

        self.train_dataset = SpeechRecognitionDatasetSimplified(self.training_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.training_data.sentence,
                                                self.tokenizer,
                                                self.DATA_ROOT, raw_transform=self.audio_transform_train,mel_transform=self.mel_transorm_train,
                                                sampling_rate=self.SAMPLE_RATE,token_length=self.MAX_PREDICTION_LENGTH, pad_token=self.PAD_TOKEN)
        
        self.valid_dataset = SpeechRecognitionDatasetSimplified(self.valid_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.valid_data.sentence,
                                                self.tokenizer,
                                                self.DATA_ROOT,mel_transform=self.mel_transorm_valid,
                                                sampling_rate=self.SAMPLE_RATE,token_length=self.MAX_PREDICTION_LENGTH, pad_token=self.PAD_TOKEN,train=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.LR,weight_decay=self.WD)
        self.steps_per_epoch = len(self.train_dataset)//(self.SAMPLES_PER_GPU*self.N_GPU)+1
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.LR,steps_per_epoch=self.steps_per_epoch,epochs=self.EPOCHS,pct_start=0.1)
        self.criterion = MaskedCrossEntropyLoss(self.PAD_TOKEN)

    def load_state_dict(self,path):
        statedict = torch.load(path)
        print("loading model checkpoint from epoch: ",statedict["current_step"])
        self.model.load_state_dict(statedict["model_state_dict"])