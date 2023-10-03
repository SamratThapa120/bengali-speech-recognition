from bengali_asr.models import Wav2Vec2WithLM
from bengali_asr.dataset.tokenizer import BengaliTokenizer
from bengali_asr.dataset.encoder_decoder_dataset import SpeechRecognitionDataset
from bengali_asr.models.loss import MaskedCrossEntropyLoss
from bengali_asr.dataset.wav2vec_dataset import SpeechRecognitionLMCollate,SpeechRecognitionCTCDataset
import pandas as pd
import torch
from .base import Base
class Configs(Base):
    OUTPUTDIR="../workdir/wav2vec2_characterlevel_pretrained_ctcloss_lm"
    TRAIN_DATA_PATH="/app/dataset/train_data_subset.csv"
    VALID_DATA_PATH="/app/dataset/valid_data_subset.csv"
    DATA_ROOT="/app/dataset/train_numpy_16k"
    
    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=12
    N_GPU=4
    ENCODER_UNFREEZE_EPOCH=10

    VALIDATION_BS=16
    VALIDATION_FREQUENCY=5000
    PIN_MEMORY=True
    NUM_WORKERS=4
    NUM_WORKERS_VAL=4
    DISTRIBUTED=True
    FREEZE_ENCODER=True
    LR=0.0005
    EPOCHS=20
    
    MAX_TOKEN_LENGTH=256
    MAX_PREDICTION_LENGTH=MAX_TOKEN_LENGTH
    MAX_AUDIO_LENGTH=163840
    AUDIO_PADDING=0.0
    TRAIN_TYPE="wav2vec_lm"
    AUTOCAST=False
    augoregressive_inference=False
    
    AUDIO_SCALE=320
    
    VOCAB = ['ও', ' ', 'ব', 'ল', 'ে', 'ছ', 'আ', 'প', 'ন', 'া', 'র', 'ঠ', 'ি', 'ক', '!', 'ো', 'ম', 'হ', 'ষ', '্', 'ট', 'গ', 'ত', 'চ', '?', 'ু', 'ঝ', ',', 'এ', 'স', 'থ', '।', 'শ', 'য', '়', 'ী', 'ধ', 'ঙ', 'ভ', 'জ', 'ই', 'দ', 'খ', 'ফ', 'ং', 'উ', 'ণ', 'অ', 'ঁ', 'ড়', 'য়', 'ঢ', 'ড', '-', 'ূ', 'ঘ', 'ৃ', 'ঞ', '‘', '’', 'ৈ', '"', '—', 'ৌ', 'ৎ', 'ঃ', ';', 'ঐ', 'ঈ', 'ঊ', '–', "'", 'ঋ', ':', '/', 'ঢ়', 'ঔ', '.', '“', '”']
    START_TOKEN=len(VOCAB)
    END_TOKEN=len(VOCAB)+1
    PAD_TOKEN=-1


    def __init__(self,inference_files=None,inference_text=None,use_numpy=False):
        self.device = "cuda"
        self.dataloder_collate = SpeechRecognitionLMCollate(self.MAX_TOKEN_LENGTH,
                                            self.MAX_AUDIO_LENGTH,
                                            self.AUDIO_PADDING,
                                            self.PAD_TOKEN,
                                            self.END_TOKEN,
                                            self.AUDIO_SCALE)
        self.model = Wav2Vec2WithLM(len(self.VOCAB)+2,
                                max_encoder_states=self.MAX_AUDIO_LENGTH//self.AUDIO_SCALE+1,
                                max_output_len=self.MAX_TOKEN_LENGTH,
                                decoder_heads=8,
                                decoder_layer=6,
                                attention_dropout=0.1, 
                                hidden_dropout=0.1, 
                                feat_proj_dropout = 0.1,
                                layerdrop=0.1,
                                pretrained="facebook/wav2vec2-xls-r-300m"  if inference_files is None else None,
                                activation_dropout=0.0,
                                mask_time_prob=0.05,
                                mask_time_length=8)
        self.tokenizer_train = BengaliTokenizer(self.VOCAB,self.START_TOKEN,self.END_TOKEN)
        self.tokenizer = BengaliTokenizer(self.VOCAB,self.START_TOKEN,self.END_TOKEN)
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
        self.audio_transform_train = None
        
        self.training_data = pd.read_csv(self.TRAIN_DATA_PATH)[:self.USE_DATASET_LEN]
        self.valid_data = pd.read_csv(self.VALID_DATA_PATH)[:self.USE_DATASET_LEN]
        print(f"length of train: {len(self.training_data)}, length of valid: {len(self.valid_data)}")

        self.train_dataset = SpeechRecognitionCTCDataset(self.training_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.training_data.sentence,
                                                self.tokenizer_train,
                                                self.DATA_ROOT,
                                                raw_transform=self.audio_transform_train,
                                                sampling_rate=self.SAMPLE_RATE)
        
        self.valid_dataset = SpeechRecognitionCTCDataset(self.valid_data.id.apply(lambda x: x.replace(".mp3",".npy")),
                                                self.valid_data.sentence,
                                                self.tokenizer,
                                                self.DATA_ROOT,
                                                sampling_rate=self.SAMPLE_RATE,
                                                train=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=self.LR)
        self.steps_per_epoch = len(self.train_dataset)//(self.SAMPLES_PER_GPU*self.N_GPU)+1
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.LR,steps_per_epoch=self.steps_per_epoch,epochs=self.EPOCHS,pct_start=0.2)
        self.criterion = MaskedCrossEntropyLoss(self.PAD_TOKEN)

    def load_state_dict(self,path):
        statedict = torch.load(path)
        print("loading model checkpoint from epoch: ",statedict["current_step"])
        self.model.load_state_dict(statedict["model_state_dict"])