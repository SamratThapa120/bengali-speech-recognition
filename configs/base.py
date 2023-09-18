from bengali_asr.audio.utils import exact_div

class Base():
    SAMPLE_RATE = 16000
    N_FFT = 400
    N_MELS = 80
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

    N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
    FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
    TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token
    TRAIN_TYPE=""
    augoregressive_inference=True
    NUM_WORKERS_VAL=4

    FREEZE_ENCODER=False
    AUTOCAST=False
    def get_all_attributes(obj):
        attributes = {}
        for key, value in vars(obj.__class__).items():
            if not key.startswith('__'):
                attributes[key] = value
        
        for key, value in vars(obj).items():
            if not key.startswith('__'):
                attributes[key] = value           
        return attributes
