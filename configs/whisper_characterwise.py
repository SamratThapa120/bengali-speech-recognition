from bengali_asr.models import ModelDimensions
from bengali_asr.models import Whisper
from bengali_asr.dataset.tokenizer import CharacterLevelTokenizer
from bengali_asr.dataset.encoder_decoder_dataset import SpeechRecognitionDataset
from bengali_asr.audio import LogMelSpectrogramTransform
from bengali_asr.dataset.transforms import ComposeAll
from bengali_asr.models.loss import MaskedCrossEntropyLoss
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .base import Base

class Configs(Base):
    OUTPUTDIR="../workdir/whisperbase_characterlevel"

    TRAIN_DATA_PATH="/app/dataset/train_data.csv"
    VALID_DATA_PATH="/app/dataset/valid_data_subset.csv"
    DATA_ROOT="/app/dataset/train_numpy_16k"
    USE_DATASET_LEN=100   #Set to small number while debugging
    SAMPLES_PER_GPU=32
    N_GPU=4
    VALIDATION_BS=32
    VALIDATION_FREQUENCY=1
    PIN_MEMORY=True
    NUM_WORKERS=4
    DISTRIBUTED=True

    LR=0.0005
    EPOCHS=50
    
    
    VOCAB = ['ও', ' ', 'ব', 'ল', 'ে', 'ছ', 'আ', 'প', 'ন', 'া', 'র', 'ঠ', 'ি', 'ক', '!', 'ো', 'ম', 'হ', 'ষ', '্', 'ট', 'গ', 'ত', 'চ', '?', 'ু', 'ঝ', ',', 'এ', 'স', 'থ', '।', 'শ', 'য', '়', 'ী', 'ধ', 'ঙ', 'ভ', 'জ', 'ই', 'দ', 'খ', 'ফ', 'ং', 'উ', 'ণ', 'অ', 'ঁ', 'ড়', 'য়', 'ঢ', 'ড', '-', 'ূ', 'ঘ', 'ৃ', 'ঞ', '‘', '’', 'ৈ', '"', '—', 'ৌ', 'ৎ', 'ঃ', ';', 'ঐ', 'ঈ', 'ঊ', '–', "'", 'ঋ', ':', '/', 'ঢ়', 'ঔ', '.', '“', '”']
    START_TOKEN=0
    END_TOKEN=len(VOCAB)+1
    MAX_PREDICTION_LENGTH=256
    PAD_TOKEN=-1

    def __init__(self):
        self.device = "cuda"
        self.training_data = pd.read_csv(self.TRAIN_DATA_PATH)[:self.USE_DATASET_LEN]
        self.valid_data = pd.read_csv(self.VALID_DATA_PATH)[:self.USE_DATASET_LEN]

        print(f"length of train: {len(self.training_data)}, length of valid: {len(self.valid_data)}")
        self.model_dims = ModelDimensions(n_mels=self.N_MELS, 
                                    n_audio_ctx=self.N_FRAMES//2, 
                                    n_audio_state=512,
                                    n_audio_head=8, 
                                    n_audio_layer=6, 
                                    n_vocab=len(self.VOCAB)+2, 
                                    n_text_ctx=448, 
                                    n_text_state=512, 
                                    n_text_head=8, 
                                    n_text_layer=6)
        self.model = Whisper(self.model_dims)
        self.tokenizer = CharacterLevelTokenizer(self.VOCAB,self.START_TOKEN,self.END_TOKEN)
        self.mel_transorm_train = ComposeAll([
            LogMelSpectrogramTransform(self.N_MELS,self.N_FFT,self.HOP_LENGTH,self.SAMPLE_RATE,tensor_length=self.N_FRAMES),
            ])
        self.mel_transorm_valid = ComposeAll([
            LogMelSpectrogramTransform(self.N_MELS,self.N_FFT,self.HOP_LENGTH,self.SAMPLE_RATE,tensor_length=self.N_FRAMES),
            ])
        self.train_dataset = SpeechRecognitionDataset(self.training_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.training_data.sentence,
                                                self.tokenizer,
                                                self.DATA_ROOT,mel_transform=self.mel_transorm_train,
                                                sampling_rate=self.SAMPLE_RATE,token_length=self.MAX_PREDICTION_LENGTH, pad_token=self.PAD_TOKEN)
        
        self.valid_dataset = SpeechRecognitionDataset(self.valid_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.valid_data.sentence,
                                                self.tokenizer,
                                                self.DATA_ROOT,mel_transform=self.mel_transorm_valid,
                                                sampling_rate=self.SAMPLE_RATE,token_length=self.MAX_PREDICTION_LENGTH, pad_token=self.PAD_TOKEN,train=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.LR)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.LR,steps_per_epoch=len(self.train_dataset)//(self.SAMPLES_PER_GPU*self.N_GPU)+1,epochs=self.EPOCHS,pct_start=0.1)
        self.criterion = MaskedCrossEntropyLoss(self.PAD_TOKEN)