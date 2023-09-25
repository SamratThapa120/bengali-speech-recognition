from bengali_asr.models import Wav2Vec2Base
from bengali_asr.dataset.tokenizer import CharacterLevelCTCTokenizer
from bengali_asr.dataset.wav2vec_dataset import SpeechRecognitionCTCDataset,SpeechRecognitionCollate
from bengali_asr.dataset.transforms import ComposeAll
from bengali_asr.models.loss import CTCLossBatchFirst
import pandas as pd
import torch
from .base import Base
from bengali_asr.dataset.waveform_augments import GaussianNoise,TimeAugment,ResampleAugmentation


class Configs(Base):
    OUTPUTDIR="../workdir/wav2vec2_characterlevel_pretrained_ctcloss_augs"
    TRAIN_DATA_PATH="/app/dataset/train_data.csv"
    VALID_DATA_PATH="/app/dataset/valid_data_subset.csv"
    DATA_ROOT="/app/dataset/train_numpy_16k"
    
    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=10
    N_GPU=4
    ENCODER_UNFREEZE_EPOCH=10

    VALIDATION_BS=32
    VALIDATION_FREQUENCY=1
    PIN_MEMORY=True
    NUM_WORKERS=4
    NUM_WORKERS_VAL=4
    DISTRIBUTED=True
    FREEZE_ENCODER=True
    LR=0.001
    EPOCHS=20

    
    
    VOCAB = ['ও', ' ', 'ব', 'ল', 'ে', 'ছ', 'আ', 'প', 'ন', 'া', 'র', 'ঠ', 'ি', 'ক', '!', 'ো', 'ম', 'হ', 'ষ', '্', 'ট', 'গ', 'ত', 'চ', '?', 'ু', 'ঝ', ',', 'এ', 'স', 'থ', '।', 'শ', 'য', '়', 'ী', 'ধ', 'ঙ', 'ভ', 'জ', 'ই', 'দ', 'খ', 'ফ', 'ং', 'উ', 'ণ', 'অ', 'ঁ', 'ড়', 'য়', 'ঢ', 'ড', '-', 'ূ', 'ঘ', 'ৃ', 'ঞ', '‘', '’', 'ৈ', '"', '—', 'ৌ', 'ৎ', 'ঃ', ';', 'ঐ', 'ঈ', 'ঊ', '–', "'", 'ঋ', ':', '/', 'ঢ়', 'ঔ', '.', '“', '”']
    BLANK_TOKEN = len(VOCAB)
    START_TOKEN=0
    END_TOKEN=len(VOCAB)+1
    MAX_TOKEN_LENGTH=256
    MAX_AUDIO_LENGTH=400000
    AUDIO_PADDING=0.0
    PAD_TOKEN=-1
    TRAIN_TYPE="wav2vec_ctc"
    AUTOCAST=False
    augoregressive_inference=False
    
    AUDIO_SCALE=320
    MAX_FEATURE_LENGTH=400000
    def __init__(self,inference_files=None,inference_text=None,use_numpy=False):
        self.device = "cuda"
        self.model = Wav2Vec2Base(len(self.VOCAB)+1)
        self.tokenizer = CharacterLevelCTCTokenizer(self.VOCAB)
        self.mel_transorm_valid = None
        if inference_files is not None:
            print("inference mode is on")
            self.inference_dataset = SpeechRecognitionCTCDataset(inference_files,
                                        inference_text,
                                        self.tokenizer,
                                        self.DATA_ROOT,
                                        sampling_rate=self.SAMPLE_RATE,
                                        train=False,
                                        usenumpy=use_numpy) 
            return
        
        #Below are the 
        self.training_data = pd.read_csv(self.TRAIN_DATA_PATH)[:self.USE_DATASET_LEN]
        self.valid_data = pd.read_csv(self.VALID_DATA_PATH)[:self.USE_DATASET_LEN]
        print(f"length of train: {len(self.training_data)}, length of valid: {len(self.valid_data)}")
        self.audio_transform_train = ComposeAll([
            ResampleAugmentation(p=0.5),
            TimeAugment(p=0.5),
            GaussianNoise(p=0.5)
        ])
        self.train_dataset = SpeechRecognitionCTCDataset(self.training_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.training_data.sentence,
                                                self.tokenizer,
                                                self.DATA_ROOT,
                                                raw_transform=self.audio_transform_train,
                                                sampling_rate=self.SAMPLE_RATE)
        
        self.valid_dataset = SpeechRecognitionCTCDataset(self.valid_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.valid_data.sentence,
                                                self.tokenizer,
                                                self.DATA_ROOT,
                                                sampling_rate=self.SAMPLE_RATE,
                                                train=False)
        self.dataloder_collate = SpeechRecognitionCollate(self.MAX_TOKEN_LENGTH,
                                                          self.MAX_AUDIO_LENGTH,
                                                          self.AUDIO_PADDING,
                                                          self.PAD_TOKEN,self.AUDIO_SCALE)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.LR)
        self.steps_per_epoch = len(self.train_dataset)//(self.SAMPLES_PER_GPU*self.N_GPU)+1
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.LR,steps_per_epoch=self.steps_per_epoch,epochs=self.EPOCHS,pct_start=0.1)
        self.criterion = CTCLossBatchFirst(blank=self.BLANK_TOKEN,ignore_index=self.PAD_TOKEN)

        self.grad_scaler = torch.cuda.amp.GradScaler()
    def load_state_dict(self,path):
        statedict = torch.load(path)
        print("loading model checkpoint from epoch: ",statedict["epoch"])
        self.model.load_state_dict(statedict["model_state_dict"])