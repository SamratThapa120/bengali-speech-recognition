from bengali_asr.models import Wav2Vec2WithLM
from bengali_asr.dataset.tokenizer import BengaliTokenizer
from bengali_asr.dataset.encoder_decoder_dataset import SpeechRecognitionDataset
from bengali_asr.models.loss import MaskedCrossEntropyLoss
from bengali_asr.dataset.wav2vec_dataset import SpeechRecognitionLMCollate,SpeechRecognitionCTCDataset
import pandas as pd
import torch
from .base import Base
class Configs(Base):
    OUTPUTDIR="../workdir/wav2vec2_characterlevel_pretrained_ctcloss_lm_newtokenizer"
    TRAIN_DATA_PATH="/app/dataset/train_data_subset.csv"
    VALID_DATA_PATH="/app/dataset/valid_data_subset.csv"
    DATA_ROOT="/app/dataset/train_numpy_16k"
    
    USE_DATASET_LEN=None   #Set to small number while debugging
    SAMPLES_PER_GPU=12
    N_GPU=4
    ENCODER_UNFREEZE_EPOCH=10

    VALIDATION_BS=16
    VALIDATION_FREQUENCY=4000
    PIN_MEMORY=True
    NUM_WORKERS=4
    NUM_WORKERS_VAL=4
    DISTRIBUTED=True
    FREEZE_ENCODER=True
    LR=0.0005
    EPOCHS=20
    
    MAX_TOKEN_LENGTH=150
    MAX_AUDIO_LENGTH=163840
    AUDIO_PADDING=0.0
    TRAIN_TYPE="wav2vec_ctc"
    AUTOCAST=False
    augoregressive_inference=False
    
    AUDIO_SCALE=320
    
    VOCAB = [' ', 'র', 'ক', 'ন', '।', 'ব', 'আ', 'এ', 'ই', 'ম', 'স', 'ত', 'ল', 'রে', 'প', 'রা', 'ন্', 'নি', 'হ', 'মা', 'কে', 'র্', 'না', 'তি', 'কা', 'বা', 'তা', 'লে', ',', 'তে', 'ও', 'অ', 'দে', 'য', 'নে', 'যা', 'বে', 'স্', 'জ', 'প্', 'বি', 'ছে', 'ক্', 'রি', 'য়', 'ত্', 'লা', 'য়ে', 'পা', 'টি', 'শ', 'সে', 'দ', 'ছি', 'সা', 'মি', 'গ', 'য়া', 'দি', 'টা', 'থা', 'ভা', 'হা', 'উ', 'ট', 'কি', 'চ', 'দ্', 'লি', 'ণ', 'ষ', 'য়ে', 'জা', 'যে', 'খ', 'ব্', 'মে', 'য়', 'গে', 'খা', 'ষ্', 'তু', 'ম্', 'দা', 'চা', 'ধ', 'তো', 'থে', 'শ্', 'শি', 'লো', '-', 'শে', 'হি', 'চ্', 'জে', '?', 'রু', 'থ', 'গ্', 'ধা', 'রী', 'চি', 'সি', 'বং', 'ল্', 'টে', 'কো', 'ঙ্', 'ভি', 'শা', 'দু', 'ধ্', 'পু', 'য়া', 'গু', 'নো', 'ভ', 'গা', 'খে', 'মু', 'ছ', 'রো', 'পে', 'নু', 'শু', 'তাঁ', 'সং', 'সু', 'নী', 'জি', 'জ্', 'ফ', 'চে', 'ঘ', 'ছু', 'ষে', 'যু', 'ষা', 'ছা', 'ড', '!', 'ট্', 'যো', 'বু', 'কু', 'ড়ি', 'ড়া', 'তী', 'গি', 'ধি', 'ৎ', 'ঠা', 'বো', 'ঠি', 'ণে', 'ডি', 'পি', 'ঞ্', 'পূ', 'ডা', 'ফি', 'ড়', 'ডে', 'জী', 'থি', 'কী', 'হে', 'বী', 'খু', 'ড়ে', 'ণা', 'বাং', 'ধু', 'দী', 'ফে', 'ঠ', 'মী', 'কৃ', 'লী', 'খি', 'ড়া', 'ধে', 'মূ', 'হ্', 'ণ্', 'মো', 'গো', 'শী', 'ষি', 'ভে', '"', 'লু', 'খ্', 'ঞা', 'টু', 'ধী', 'সী', 'ভূ', 'হু', 'ফা', 'বৃ', 'ছো', 'চু', 'অং', 'শো', 'গী', 'তৃ', 'ঠে', 'রূ', 'ভু', 'সো', 'জু', 'হী', 'মৃ', 'অ্', 'সূ', 'দো', 'ফ্', 'পো', 'ফু', 'ঐ', 'ণি', 'ঢা', 'ণী', 'থ্', 'বৈ', 'ভ্', 'তৈ', 'ড়ে', 'লো', 'দূ', 'য়ো', 'হো', 'নো', 'টো', 'ড়ি', 'ঝ', 'ঘু', 'য়ি', 'ঘা', 'যি', 'ফো', 'খো', 'চো', 'যাঁ', 'বাঁ', 'থী', 'ভো', 'জো', 'ইং', 'ড়', 'চী', 'পাঁ', 'দাঁ', 'দৃ', 'ঝে', 'পৃ', 'পী', 'ঝা', 'নৈ', 'ঈ', 'য়ী', 'কো', 'ঘো', 'য্', 'ড্', 'খুঁ', 'ষু', 'রো', 'ভী', 'য়ো', 'যাং', 'যো', 'সৃ', "'", 'ঝি', 'য়ি', 'টিং', 'তো', 'কিং', 'সিং', 'য়ী', 'সাং', 'দুঃ', 'লিং', 'ঞ', 'হৃ', 'ঙে', 'কাঁ', 'মৌ', 'ড়ো', 'রং', 'তঃ', 'পৌঁ', 'কৌ', 'ষী', 'সৈ', '–', 'ঙা', '’', 'চৌ', 'গৃ', 'বেঁ', 'পৌ', 'শৈ', 'ডু', 'ঢু', 'ফাঁ', 'সৌ', 'বৌ', 'হাঁ', 'ঙ', '—', 'যৌ', 'কাং', 'নৌ', 'খী', 'হূ', 'ডো', 'কূ', 'ঘ্', 'মো', 'রাং', 'য়ো', 'নিং', 'ঠো', 'আঁ', 'ঠী', 'দৈ', 'খোঁ', 'শূ', 'বো', 'ণু', '‘', 'চাঁ', 'শং', 'নূ', 'ঋ', 'ড়ী', 'কং', 'ঘে', 'দৌ', 'ঝুঁ', 'মাং', 'ষো', 'য়ু', 'নৃ', 'গো', 'ঞে', 'চূ', 'পো', 'শৃ', 'নিঃ', 'গাঁ', 'দো', 'ঘি', 'ধূ', 'হিং', 'ডিং', 'গৌ', 'শো', ':', 'ভো', 'রিং', 'য়ু', 'ঝু', 'সো', 'ঢে', 'ভৃ', 'হো', 'ফো', 'ঔ', 'সাঁ', 'ঢ়', 'ঠু', 'ঝাঁ', ';', 'এঁ', 'ভৌ', 'ড়ী', 'খাঁ', 'জো', 'য়ং', 'ঙি', 'টো', 'ঘৃ', 'জৈ', 'উঁ', 'থু', 'ছো', 'হং', 'ঙু', 'পুঁ', 'ঘো', 'য়্', 'ঊ', 'টী', 'চো', 'বিং', 'হেঁ', '.', 'মিং', 'নঃ', 'ঝো', 'ডী', 'লং', 'থো', 'ড়ু', 'আং', 'ইঁ', 'ছুঁ', 'ধো', 'ওঁ', 'তূ', 'চৈ', 'রাঁ', 'সিঁ', 'খো', 'ঠো', 'ধৈ', 'ঘাঁ', 'কৈ', 'ঢো', '/', 'চিং', 'য়্', 'ঠ্', 'গুঁ', 'জং', 'শিং', 'পেঁ', 'জিং', 'পিং', 'ঠোঁ', 'ড়ু', 'রৌ', 'খোঁ', 'লৌ', 'গোঁ', 'পৈ', 'ছোঁ', 'ড়ো', 'যী', 'য়াং', 'ছিঁ', 'য়ং', 'কেঁ', 'ভাঁ', 'ধোঁ', 'ছাঁ', 'যূ', 'ঘেঁ', 'ষো', 'ঘূ', 'থূ', 'ফী', 'ধাঁ', 'শৌ', 'মং', 'ঢ়', 'ধৃ', 'ফৌ', 'ফোঁ', 'য়াং', 'মিঃ', 'ঢি', 'মৈ', 'ছ্', 'পঁ', 'যাঁঁ', 'কুঁ', 'ণো', 'নীং', 'ছেঁ', 'ভিং', 'হৈ', 'নোং', 'ভৈ', 'তাে', 'তাং', 'য়ূ', 'ঢ', 'ফুঁ', 'হিঃ', 'জাং', 'ণাং', 'ফূ', 'গেঁ', 'শুঁ', 'হুঁ', 'ঝোঁ', 'তৌ', 'ভূঁ', 'চেঁ', 'পুং', 'ঢ্', 'ঠাঁ', 'ডো', 'ফাং', 'ধঃ', 'ঢো', 'গৈ', 'ঝো', 'দং', '“', 'ডঃ', 'য়ূ', 'এ্', 'পাং', 'ছেঃ', 'লাে', 'রৈ', 'লূ', '”', 'থাং', 'শঃ', 'তঁা', 'শাঁ', 'বঁ', 'সেঁ', 'য়িং', 'আঃ', 'পিঁ', 'লঃ', 'খৃ', 'লাং', 'সঁ', 'ফেঁ', 'ড়ো', 'হিঁ', 'বিঁ', 'জৌ', 'গূ', 'টাং', 'ধো', 'ষাং', 'রোং', 'নাং', 'কোঁ', 'যিং', 'ঢং', 'জাঁ', 'পং', 'তেঁ', 'তং', 'ফেং', 'ষাঁ', 'রাে', 'যং', 'ভোঁ', 'সোঁ', 'পোঁ', 'ধৌ', 'মাে', 'লোঃ', 'ঞী', 'খিঁ', 'লৈ', 'ফিং', 'ঢুঁ', 'ভুঁ', 'হঃ', 'গং', 'বাঃ', 'টেং', 'লেং', 'শাে', 'নং', 'চৈঃ', 'হাং', 'লুং', 'ধুঁ', 'কৌঁ', 'ঘী', 'ড়্', 'থো', 'নাে', 'টিঁ', 'চুঁ', 'ঢ়া', 'ছোঁ', 'ডাঃ', 'ঢৌ', 'য়ঃ', 'বাে', 'রুং', 'শাং', 'দৌঁ', 'থেঁ', 'চং', 'যৈ', 'য়ঃ', 'চিঁ', 'হোঁ', 'গাং', 'মঃ', 'রেং', 'চাং', 'য়িং', 'ওং', 'টং', 'গাে', 'ভাং', 'তুং', 'ধোঁ', 'ধাং', 'ঘুঁ', 'রঃ', 'নোং', 'জেঁ', 'তিঃ', 'সোং', 'টুঁ', 'সেং', 'কোং', 'ড়ং', 'ঢ়ে', 'জঃ', 'ঢেঁ', 'খৈ', 'ভেং', 'ঠোঁ', 'বূ', 'নাঃ', 'আা', 'ঝং', 'টূ', 'গিং', 'যাে', 'বোঁ', 'বুঁ', 'তোঁ', 'অঁ', 'হাঃ', 'কুং', 'থৈ', 'ঝেঁ', 'গোং', 'জুঁ', 'ড়ং', 'ৰ', 'যুং', 'লাঁ', 'অাঁ', 'গিঁ', 'ধং', 'থাঃ', 'ছৃ', 'ডং', 'য়াে', 'ঝ্', 'হুং', 'ণং', 'কঁ', 'রঁ', 'ঠাং', 'ণৌ', 'সুঁ', 'ছুং', 'গোঁ', 'লোং', 'রোঁ', 'ঞি', 'টেঁ', 'ডাঁ', 'ঝাং', 'উঃ', 'ঝিঁ', 'টুং', 'ৎে', 'ঘাে', 'জঁ', 'হোঁ', 'মঁ', 'সোঁ', 'ছিঃ', 'দুঁ', 'দোং', 'কুু', 'তুঁ', ' ি', 'দঁ', 'ওঃ', 'য়ঁ', 'উং', 'ফোঁ', 'য়াঁ', 'কোঁ', 'গুং', 'পঃ', '৷', 'চঁ', 'ভোঁ', 'আঁঁ', 'দঃ', 'ভং', 'চুং', 'জোঁ', 'শৃং', 'ছী', 'ছৌ', 'হোঃ', 'মোং', 'ছাে', 'যোং', 'টিা', 'হঁ', 'শোঁ', 'চোঁ']
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
        print("loading model checkpoint from epoch: ",statedict["epoch"])
        self.model.load_state_dict(statedict["model_state_dict"])