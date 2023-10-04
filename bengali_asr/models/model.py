import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math

from transformers import Wav2Vec2Model,Wav2Vec2Config

default_wav2vec2configs = {
    "activation_dropout": 0.0,
    "apply_spec_augment": True,
    "architectures": [
      "Wav2Vec2ForPreTraining"
    ],
    "attention_dropout": 0.1,
    "bos_token_id": 1,
    "codevector_dim": 768,
    "contrastive_logits_temperature": 0.1,
    "conv_bias": True,
    "conv_dim": [
      512,
      512,
      512,
      512,
      512,
      512,
      512
    ],
    "conv_kernel": [
      10,
      3,
      3,
      3,
      3,
      2,
      2
    ],
    "conv_stride": [
      5,
      2,
      2,
      2,
      2,
      2,
      2
    ],
    "ctc_loss_reduction": "sum",
    "ctc_zero_infinity": False,
    "diversity_loss_weight": 0.1,
    "do_stable_layer_norm": True,
    "eos_token_id": 2,
    "feat_extract_activation": "gelu",
    "feat_extract_dropout": 0.0,
    "feat_extract_norm": "layer",
    "feat_proj_dropout": 0.1,
    "feat_quantizer_dropout": 0.0,
    "final_dropout": 0.0,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-05,
    "layerdrop": 0.1,
    "mask_feature_length": 10,
    "mask_feature_prob": 0.0,
    "mask_time_length": 10,
    "mask_time_prob": 0.075,
    "model_type": "wav2vec2",
    "num_attention_heads": 16,
    "num_codevector_groups": 2,
    "num_codevectors_per_group": 320,
    "num_conv_pos_embedding_groups": 16,
    "num_conv_pos_embeddings": 128,
    "num_feat_extract_layers": 7,
    "num_hidden_layers": 24,
    "num_negatives": 100,
    "pad_token_id": 0,
    "proj_codevector_dim": 768,
    "torch_dtype": "float32",
    "transformers_version": "4.12.0.dev0",
    "use_weighted_layer_sum": False
}  
class Wav2Vec2Base(torch.nn.Module):
    def __init__(self, vocab_size,attention_dropout=0.1, hidden_dropout=0.1, feat_proj_dropout = 0.1,
                    mask_time_prob=0,layerdrop=0,classifier_dropout=0.1,pretrained="facebook/wav2vec2-xls-r-300m",**kwargs):
        super().__init__()
        if pretrained is not None:
            self.model = Wav2Vec2Model.from_pretrained(
                pretrained, 
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                feat_proj_dropout=feat_proj_dropout,
                mask_time_prob=mask_time_prob,
                layerdrop=layerdrop,**kwargs)
        else:
            default_configs = default_wav2vec2configs.copy()
            default_configs.update({
                "attention_dropout":attention_dropout,
                "hidden_dropout":hidden_dropout,
                "feat_proj_dropout":feat_proj_dropout,
                "mask_time_prob":mask_time_prob,
                "layerdrop":layerdrop
            })
            default_configs.update(kwargs)
            self.model = Wav2Vec2Model(Wav2Vec2Config(**default_configs))
        self.dropout = torch.nn.Dropout(p=classifier_dropout)
        self.classifier = torch.nn.Linear(1024,vocab_size)

    def forward(self,inp):
        return self.classifier(self.dropout(self.model(inp).last_hidden_state))

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    n_max_len: int = 256
    use_mask: bool = True
    use_lm: bool = True

class Wav2Vec2WithLM(torch.nn.Module):
    def __init__(self, vocab_size,max_encoder_states,max_output_len:int=256,decoder_heads:int=8,decoder_layer:int=6,attention_dropout=0.1, hidden_dropout=0.1, feat_proj_dropout = 0.1,
                    mask_time_prob=0,layerdrop=0,pretrained="facebook/wav2vec2-xls-r-300m",**kwargs):
        super().__init__()
        if pretrained is not None:
            self.encoder = Wav2Vec2Model.from_pretrained(
                pretrained, 
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                feat_proj_dropout=feat_proj_dropout,
                mask_time_prob=mask_time_prob,
                layerdrop=layerdrop,**kwargs)
        else:
            default_configs = default_wav2vec2configs.copy()
            default_configs.update({
                "attention_dropout":attention_dropout,
                "hidden_dropout":hidden_dropout,
                "feat_proj_dropout":feat_proj_dropout,
                "mask_time_prob":mask_time_prob,
                "layerdrop":layerdrop
            })
            default_configs.update(kwargs)
            self.encoder = Wav2Vec2Model(Wav2Vec2Config(**default_configs))
        self.decoder = TextDecoder(vocab_size,max_encoder_states,1024,decoder_heads,decoder_layer)

    def forward(self,audio,tokens):
        return self.decoder(tokens,self.encoder(audio).last_hidden_state)



import torch.nn as nn
from torch import Tensor

class LayerNorm(nn.LayerNorm):
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        # The eps value is already used in the parent class's forward method.
        # So, there's no need to use it explicitly here.
        if x.dtype == torch.float16:
            return super().forward(x.float()).type(x.dtype)
        else:
            return super().forward(x)



class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx*2)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # if not self.training:
        n_ctx_cur = x.shape[1]
        positional_embedding_cur = self.positional_embedding[:n_ctx_cur, :]
        # else:
            # positional_embedding_cur = self.positional_embedding
        # assert x.shape[1:] == positional_embedding_cur.shape, "incorrect audio shape"

        x = (x + positional_embedding_cur)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.nn.init.normal_(torch.rand(n_ctx, n_state)))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class TextDecoderNoLM(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, max_len:int=256,dropout: float=0.1,use_mask: bool = True
    ):
        super().__init__()
        print("Using ext decoder without language model")
        
        # Initialize query embeddings
        self.query_embeddings = nn.Parameter(torch.randn(max_len, n_state))
        
        # Compute and store positional encodings
        print("Using positional embedding")
        self.positional_encodings = positionalencoding1d(n_state, max_len)
        self.register_buffer("pos_enc", self.positional_encodings)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(n_state,n_vocab)

        if use_mask:
            mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
            self.register_buffer("mask", mask, persistent=False)
        else:
            mask=None

    def forward(self, tokens: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        # Add positional encodings to the query embeddings
        x = self.query_embeddings
        x = x.unsqueeze(0).expand(xa.size(0), -1, -1)
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x+self.pos_enc, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = self.fc(self.dropout(x))
        
        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        if self.dims.use_lm:
           self.decoder = TextDecoder(
                self.dims.n_vocab,
                self.dims.n_text_ctx,
                self.dims.n_text_state,
                self.dims.n_text_head,
                self.dims.n_text_layer,
            )         
        else:
           self.decoder = TextDecoderNoLM(
                self.dims.n_vocab,
                self.dims.n_text_ctx,
                self.dims.n_text_state,
                self.dims.n_text_head,
                self.dims.n_text_layer,
                self.dims.n_max_len,
                use_mask=self.dims.use_mask
            )   

        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads, persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask, persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor=None
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks