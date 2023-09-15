import torch
from whisper.tokenizer import get_tokenizer

class CharacterLevelTokenizer:
    def __init__(self,characters : list,start_token=0,end_token=-1) -> None:
        characters = sorted(characters)
        characters.insert(start_token,"<start>")
        characters.insert(end_token,"<end>")
        self.chars = {c:i for i,c in enumerate(characters)}
        self.idx_to_chars = {i:c for i,c in enumerate(characters)}

        self.start_token = start_token
        self.end_token = end_token if end_token!=-1 else len(characters)

    def __call__(self,transcript: str,add_extras=True):
        if add_extras:
            encoded = [self.start_token]
        else:
            encoded=[]
        for i in transcript:
            if i in self.chars:
                encoded.append(self.chars[i])

        if add_extras:
            encoded.append(self.end_token)
        return torch.tensor(encoded)
    
    def decode(self,tokens):
        return "".join([self.idx_to_chars[i] for i in tokens])
    
    def decode_torch_inference(self,tokens):
        tokens = tokens.detach().cpu().numpy()
        return "".join([self.idx_to_chars[i] for i in tokens])
    
class CharacterLevelCTCTokenizer:
    def __init__(self,characters : list) -> None:
        characters = sorted(characters)
        self.chars = {c:i for i,c in enumerate(characters)}
        self.idx_to_chars = {i:c for i,c in enumerate(characters)}

    def __call__(self,transcript: str,add_extras=True):
        encoded=[]
        for i in transcript:
            if i in self.chars:
                encoded.append(self.chars[i])
        return torch.tensor(encoded)
    
    def decode(self,tokens):
        return "".join([self.idx_to_chars[i] for i in tokens])
    
    def decode_torch_inference(self,tokens):
        tokens = tokens.detach().cpu().numpy()
        return "".join([self.idx_to_chars[i] for i in tokens])

class WhisperTokenizer:

    def __init__(self,vocabulary) -> None:
        self.tokenizer = tknizer = get_tokenizer(multilingual=True)

        self.chars = {k:self.tokenizer.encode(k) for k in vocabulary}

        self.start_token = tknizer.sot
        self.end_token = tknizer.eot

    def __call__(self,transcript: str,add_extras=True):
        if add_extras:
            encoded = [self.start_token]
        else:
            encoded=[]
        for i in transcript:
            if i in self.chars:
                encoded.append(self.chars[i])

        if add_extras:
            encoded.append(self.end_token)
        return torch.tensor(encoded)
    
    def decode(self,tokens):
        return "".join([self.idx_to_chars[i] for i in tokens])
    
    def decode_torch_inference(self,tokens):
        tokens = tokens.detach().cpu().numpy()
        return "".join([self.idx_to_chars[i] for i in tokens])
