from bengali_asr.models import Wav2Vec2WithLM
from bengali_asr.dataset.tokenizer import BengaliTokenizer
from bengali_asr.dataset.encoder_decoder_dataset import SpeechRecognitionDataset
from bengali_asr.models.loss import MaskedCrossEntropyLoss
from bengali_asr.dataset.wav2vec_dataset import SpeechRecognitionLMCollate,SpeechRecognitionCTCDataset
import pandas as pd
import torch
from .base import Base
class Configs(Base):
    OUTPUTDIR="../workdir/wav2vec2_characterlevel_pretrained_ctcloss_lm_openasr"
    TRAIN_DATA_PATH="/app/dataset/train_data_with_openasr.csv"
    VALID_DATA_PATH="/app/dataset/valid_data_subset.csv"
    DATA_ROOT="/app/dataset/train_numpy_16k"
    
    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=12
    N_GPU=4
    ENCODER_UNFREEZE_EPOCH=1

    VALIDATION_BS=16
    VALIDATION_FREQUENCY=15000
    PIN_MEMORY=True
    NUM_WORKERS=4
    NUM_WORKERS_VAL=4
    DISTRIBUTED=True
    FREEZE_ENCODER=True
    LR=0.0005
    EPOCHS=20
    
    MAX_TOKEN_LENGTH=150
    MAX_PREDICTION_LENGTH=MAX_TOKEN_LENGTH
    MAX_AUDIO_LENGTH=163840
    AUDIO_PADDING=0.0
    TRAIN_TYPE="wav2vec_lm"
    AUTOCAST=False
    augoregressive_inference=False
    
    AUDIO_SCALE=320
    
    VOCAB = [' ', 'র', 'ক', 'ন', '।', 'ব', 'আ', 'এ', 'ই', 'ম', 'স', 'ত', 'ল', 'রে', 'প', 'রা', 'হ', 'ন্', 'নি', 'মা', 'কে', 'র্', 'না', 'বা', 'কা', 'তি', 'তা', 'লে', ',', 'তে', 'ও', 'অ', 'দে', 'নে', 'য', 'যা', 'বে', 'স্', 'জ', 'ছে', 'প্', 'বি', 'ক্', 'রি', 'য়', 'ত্', 'লা', 'পা', 'টি', 'য়ে', 'শ', 'দ', 'সে', 'ছি', 'সা', 'মি', 'গ', 'দি', 'টা', 'উ', 'থা', 'ভা', 'হা', 'য়া', 'ট', 'কি', 'য়ে', 'চ', 'দ্', 'য়', 'লি', 'ণ', 'ষ', 'জা', 'খ', 'যে', 'ব্', 'মে', 'গে', 'খা', 'তু', 'ষ্', 'ম্', 'দা', 'ধ', 'চা', 'তো', 'থে', 'লো', 'শ্', 'শি', '-', 'শে', 'হি', '?', 'চ্', 'জে', 'রু', 'থ', 'গ্', 'ধা', 'য়া', 'রী', 'চি', 'সি', 'বং', 'কো', 'ল্', 'টে', 'ঙ্', 'ভি', 'শা', 'দু', 'পু', 'ধ্', 'নো', 'গা', 'ভ', 'গু', 'খে', 'মু', 'ছ', 'রো', 'পে', 'নু', 'শু', 'সং', 'সু', 'তাঁ', 'নী', 'জি', 'ফ', 'জ্', 'ঘ', 'চে', 'ছু', 'ষে', 'যু', 'ষা', 'ছা', 'ড', '!', 'ট্', 'যো', 'বু', 'কু', 'তী', 'বো', 'ধি', 'ৎ', 'ড়ি', 'গি', 'ঠি', 'ঠা', 'ড়া', 'পি', 'ডি', 'ণে', 'ঞ্', 'ডা', 'পূ', 'ফি', 'ডে', 'জী', 'থি', 'কী', 'ড়', 'বী', 'হে', 'খু', 'বাং', 'ধু', 'দী', 'ণা', 'ড়া', 'ফে', 'ঠ', 'ড়ে', 'কৃ', 'মী', 'লী', 'খি', 'মো', 'মূ', 'হ্', 'গো', 'ণ্', 'ধে', 'শী', 'ষি', 'ভে', '"', 'খ্', 'লু', 'টু', 'ঞা', 'ধী', 'সী', 'ভূ', 'হু', 'ছো', 'ফা', 'বৃ', 'চু', 'শো', 'অং', 'গী', 'তৃ', 'ঠে', 'রূ', 'ভু', 'সো', 'জু', 'মৃ', 'ড়ে', 'হী', 'অ্', 'দো', 'ড়ি', 'পো', 'সূ', 'ফ্', 'ঐ', 'ফু', 'ঢা', 'ণী', 'ণি', 'ড়', 'থ্', 'বৈ', 'ভ্', 'তৈ', 'হো', 'টো', 'দূ', 'য়ি', 'ঘু', 'ঝ', 'য়ো', 'ঘা', 'যি', 'ফো', 'খো', 'চো', 'ভো', 'জো', 'লো', 'বাঁ', 'যাঁ', 'চী', 'নো', 'ইং', 'থী', 'দাঁ', 'পাঁ', 'পৃ', 'দৃ', 'ঝে', 'পী', 'ঝা', 'নৈ', 'ঈ', 'য়ো', 'ঘো', 'ড্', 'য্', 'ষু', 'ভী', 'খুঁ', 'য়ী', "'", 'সৃ', 'যাং', '১', 'য়ী', 'ঝি', 'কো', 'টিং', 'সাং', 'রো', 'কিং', 'সিং', 'য়ি', 'দুঃ', 'লিং', 'যো', 'হৃ', 'ঞ', 'ঙে', 'কাঁ', 'মৌ', '’', 'তো', 'রং', 'ড়ো', 'তঃ', 'পৌঁ', 'ষী', '০', 'কৌ', 'সৈ', '২', 'ঙা', 'চৌ', '–', 'গৃ', 'বেঁ', 'শৈ', 'পৌ', 'ডু', 'ঢু', 'হাঁ', 'সৌ', 'ফাঁ', 'ঙ', 'বৌ', 'যৌ', 'নৌ', 'কাং', 'খী', 'ডো', 'কূ', 'ঘ্', 'হূ', 'আঁ', 'ঠো', '—', 'নিং', 'রাং', 'দৈ', 'খোঁ', '৯', '‘', 'ঠী', 'শূ', 'ণু', 'চাঁ', 'শং', 'ড়ী', 'মো', 'ঋ', 'নূ', 'য়ো', 'ষো', 'দৌ', 'ঘে', 'য়ু', 'কং', 'মাং', '৩', '৫', 'ঝুঁ', 'ঞে', '৪', 'নৃ', 'বো', 'চূ', 'নিঃ', 'গাঁ', 'ধূ', 'হিং', 'শৃ', 'ঘি', ':', '৮', 'গৌ', 'ডিং', 'রিং', 'গো', '৬', 'ঝু', '.', 'পো', 'ঢে', '৭', 'ভৃ', 'য়ু', 'ঔ', 'দো', 'শো', 'ভো', 'সাঁ', 'ঠু', 'সো', 'ঢ়', 'ঝাঁ', ';', 'ভৌ', 'এঁ', 'খাঁ', 'হো', 'ফো', 'উঁ', 'য়ং', 'ড়ী', 'ঘৃ', 'ঙি', 'জৈ', 'থু', 'য়্', 'পুঁ', 'হং', 'ঙু', 'ঊ', 'জো', 'টী', 'বিং', 'লং', 'হেঁ', 'ঝো', 'ডী', 'টো', 'নঃ', 'মিং', 'চৈ', 'ছো', 'চো', 'ইঁ', 'ছুঁ', 'থো', 'আং', 'ওঁ', 'ধো', 'সিঁ', 'ঘো', 'তূ', '/', 'ড়ু', 'ঢো', 'রাঁ', 'কৈ', 'ঘাঁ', 'ড়ো', 'ধৈ', 'চিং', 'ঠ্', 'গুঁ', 'জং', 'পেঁ', 'খো', 'ড়ু', 'শিং', 'ঠো', 'ঠোঁ', 'য়্', 'জিং', 'পিং', 'রৌ', 'তাং', 'লৌ', 'ছোঁ', 'য়ং', 'য়াং', 'গোঁ', 'পৈ', 'থূ', 'কেঁ', 'ঢ়', 'যী', 'ছিঁ', 'যূ', 'ভাঁ', 'ছাঁ', 'খোঁ', 'ঘূ', 'ধোঁ', 'ধাঁ', 'ঘেঁ', 'ফী', 'মিঃ', 'ধৃ', 'মং', 'ফোঁ', 'শৌ', 'ছ্', 'পঁ', 'ফৌ', 'মৈ', 'ঢি', 'কুঁ', 'য়াং', 'হৈ', 'যাঁঁ', 'ভিং', 'ষো', 'ণো', 'নোং', 'ঢ', 'ভৈ', 'ফুঁ', 'নীং', 'ছেঁ', 'তাে', 'য়ূ', 'ণাং', 'গেঁ', 'হুঁ', '\x93', 'ঠাঁ', 'ফূ', 'চেঁ', 'জাং', 'শুঁ', 'হিঃ', 'এ্', 'ঝোঁ', 'য়ূ', 'তৌ', 'পিঁ', 'ধঃ', 'ভূঁ', 'ফাং', 'গৈ', 'ঢ্', 'পুং', 'রৈ', '\x94', 'ডঃ', '“', 'দং', 'পাং', 'শঃ', 'লাে', 'ছেঃ', '”', 'আঃ', 'ঝো', 'শাঁ', 'থাং', 'ঢো', 'লূ', 'ডো', 'বঁ', 'সেঁ', 'খৃ', 'য়িং', 'তেঁ', 'তং', 'টাং', 'তঁা', 'লঃ', 'লাং', 'সঁ', 'ফেঁ', 'গূ', 'e', 'হিঁ', 'ঢং', 'বিঁ', 'ষাং', '\u200c্', 'রোং', 'কোঁ', 'ষাঁ', '\u200d্', 'ভাং', 'পং', 'জৌ', 'নাং', 'জাঁ', 'যিং', 'ফেং', 'ড়ো', 'ভোঁ', 'রাে', 'ধো', 'r', 'ফিং', 'o', 'l', 'পোঁ', 'য়িং', 'i', 'ধৌ', 'মাে', 'যং', 'লোঃ', 'খিঁ', 'ঞী', 'ভুঁ', 'সোঁ', 'লৈ', 'গাং', 'a', 'কৌঁ', 'হাং', 'চিঁ', 'গং', 't', 'নং', 'ঢুঁ', 'টিঁ', 'লেং', 'বাঃ', 'শাে', 'হঃ', 'ঘী', 'টেং', 'ডাঃ', 'লুং', 'নাে', '%', 'দৌঁ', 'য়ঃ', 'মঃ', 'চুঁ', 'ঢ়া', 'p', 'যৈ', 'চৈঃ', 'n', 'বাে', 'রুং', 'শাং', 'ড়্', 'থেঁ', 's', 'd', 'হোঁ', 'ঢৌ', 'চাং', 'ধুঁ', 'য়ঃ', 'থো', 'চং', 'u', 'ধাং', 'রেং', 'তুং', 'টুঁ', 'ছোঁ', 'গাে', 'বোঁ', 'টং', 'ওং', 'অঁ', 'জঃ', 'আা', 'ঢ়ে', 'রঃ', '2', 'জুঁ', 'সেং', 'ঢেঁ', 'খৈ', 'তিঃ', 'জেঁ', 'f', '4', 'টূ', 'নাঃ', 'তোঁ', 'ঝং', 'গিং', 'কোং', 'ভেং', 'যাে', 'বূ', 'হাঃ', '0', 'চোঁ', 'বুঁ', 'ঠোঁ', 'ধোঁ', 'ঘুঁ', 'শৃং', 'তুঁ', '…', 'থৈ', 'ঝেঁ', 'সোং', 'ড়ং', 'নোং', '1', 'ঝিঁ', 'ডাং', 'ড়ং', 'অাঁ', 'থাঃ', 'ৰ', 'w', 'কঁ', 'শোঁ', 'ছৃ', 'কুং', 'গিঁ', 'গোং', 'য়াে', 'v', 'c', 'ঝাং', 'ণৌ', 'লাঁ', 'ধং', 'ঠাং', 'ঝ্', 'হুং', 'ডং', '3', 'টুং', 'উঃ', 'যুং', 'রোঁ', 'রঁ', 'ছুং', 'g', 'জোঁ', 'চঁ', 'ডাঁ', 'ঞি', 'দুঁ', 'b', 'ছী', 'মোং', 'ণং', 'দোং', 'সুঁ', 'জঁ', 'ৎে', ' ি', 'ওঃ', 'মঁ', 'ঘাে', 'ছিঃ', 'গোঁ', 'মোঃ', 'ফৈ', 'তঁ', 'টিা', 'গুং', '৷', 'টেঁ', 'কুু', 'দঃ', 'সোঁ', 'উং', 'ছৌ', 'হোঃ', 'ডৌ', 'h', 'নিঁ', 'ফোঁ', 'লোং', 'পঃ', 'হোঁ', 'দঁ']
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