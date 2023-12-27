import logging
from dataclasses import dataclass, asdict
from typing_extensions import Self
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# llama 2 models
llama_configs = {
    '3B': dict(n_layers=16, n_heads=32, dim=4096), # for RM model
    '7B': dict(n_layers=32, n_heads=32, dim=4096),
    '13B': dict(n_layers=40, n_heads=40, dim=5120),
    '70B': dict(n_layers=80, n_heads=64, dim=8192),
    '7B-chat': dict(n_layers=32, n_heads=32, dim=4096),
    '13B-chat': dict(n_layers=40, n_heads=40, dim=5120),
    '70B-chat': dict(n_layers=80, n_heads=64, dim=8192),
}

supported_model_types = set(llama_configs.keys())


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    multiple_of: int = 256 # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    max_batch_size: int = 8
    max_seq_len: int = 2048
    
    head_type: str = 'lm_head' # "lm_head", "scalar_head"
    use_cache: bool = False # should only use cache when do inference
    
    # dropout regularization
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    
    def __post_init__(self):
        assert self.head_type in ('lm_head', 'scalar_head')
        
    def dict(self):
        return {k: str(v) if not isinstance(v, (float, int, bool, type(None))) else v for k, v in asdict(self).items()}
    
    @classmethod
    def from_model_type(cls, model_type: str, **kwargs) -> Self:
        assert model_type in supported_model_types
        
        config = llama_configs[model_type]
        config.update(kwargs)
        
        return cls(**config)
    
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)    

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        
        self.use_cache = args.use_cache
        
        self.cache_k = None
        self.cache_v = None
        if self.use_cache:
            self.cache_k = torch.zeros((
                self.max_batch_size,
                self.max_seq_len,
                self.n_heads,
                self.head_dim
            ))
            self.cache_v = torch.zeros((
                self.max_batch_size,
                self.max_seq_len,
                self.n_heads,
                self.head_dim
            ))
            
        # regularization
        self.attn_dropout = nn.Dropout(args.attn_dropout) if args.attn_dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(args.resid_dropout) if args.resid_dropout > 0 else nn.Identity()
        
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        
    

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        resid_dropout: Optional[float] = 0.0
    ):
        super().__init__()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            resid_dropout=args.resid_dropout
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freq_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freq_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    