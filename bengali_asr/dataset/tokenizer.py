import torch

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
