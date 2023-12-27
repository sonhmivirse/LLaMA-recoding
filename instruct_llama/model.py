import logging
from dataclasses import dataclass
from typing import Optional

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
    
    head_type: str = 'lm_head'
    use_cache: bool = False
    
    