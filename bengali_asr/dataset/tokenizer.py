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
    
class CharacterLevelCTCTokenizer:
    def __init__(self,characters : list) -> None:
        characters = characters
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

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, value):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.value = value

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.value if node.is_end_of_word else None

class BengaliTokenizer:
    def __init__(self, vocabulary,start_token,end_token):
        self.trie = Trie()
        self.vocabulary = {i:c for i,c in enumerate(vocabulary)} 
        for idx, word in enumerate(vocabulary):
            self.trie.insert(word, idx)
        self.start_token=start_token
        self.end_token=end_token

    def __call__(self, sentence,add_extras=False):
        indices = []
        if add_extras:
            indices.append(self.start_token)
        i = 0
        while i < len(sentence):
            max_len = -1
            token_idx = -1
            for j in range(i, len(sentence) + 1):
                idx = self.trie.search(sentence[i:j])
                if idx is not None:
                    if j - i > max_len:
                        max_len = j - i
                        token_idx = idx
            if max_len == -1:
                # indices.append(-1)
                i += 1
            else:
                indices.append(token_idx)
                i += max_len
        if add_extras:
            indices.append(self.end_token)
        return torch.tensor(indices)

    def decode(self, indices):
        return ''.join(self.vocabulary[idx] for idx in indices if idx in self.vocabulary)
    def decode_torch_inference(self,tokens):
        tokens = tokens.detach().cpu().numpy()
        return ''.join(self.vocabulary[idx] for idx in tokens if idx in self.vocabulary)

class BengaliTokenizerCTC:
    def __init__(self, vocabulary):
        self.trie = Trie()
        self.vocabulary = {i:c for i,c in enumerate(vocabulary)} 
        for idx, word in enumerate(vocabulary):
            self.trie.insert(word, idx)

    def __call__(self, sentence,add_extras=False):
        indices = []
        i = 0
        while i < len(sentence):
            max_len = -1
            token_idx = -1
            for j in range(i, len(sentence) + 1):
                idx = self.trie.search(sentence[i:j])
                if idx is not None:
                    if j - i > max_len:
                        max_len = j - i
                        token_idx = idx
            if max_len == -1:
                # indices.append(-1)
                i += 1
            else:
                indices.append(token_idx)
                i += max_len
        return torch.tensor(indices)

    def decode(self, indices):
        return ''.join(self.vocabulary[idx] for idx in indices if idx in self.vocabulary)
    def decode_torch_inference(self,tokens):
        tokens = tokens.detach().cpu().numpy()
        return ''.join(self.vocabulary[idx] for idx in tokens if idx in self.vocabulary)