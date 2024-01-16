"""Run supervised fine-tuning (SFT) using QLoRA, starting with pretrained model."""

import torch
import torch.nn.functional as F

# support running without installing as a package
from pathlib import Path
import sys
import numpy as np

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.utils.logging import create_logger
from instruct_llama.configs.sft_lora import config as cfg

logger = create_logger()


def clear_gpu_cache(rank = None):
    torch.cuda.empty_cache()
    
def compute_finetune_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3 # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == len(mask.shape) == 2 # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0] == mask.shape[0]
    
    B, T, *_ = logits.shape
    
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
    
    assert not torch.any(torch.isnan(loss))
    
    loss = loss.view(B, T)
    
    assert loss.shape == mask.shape
    
    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # note here prompt is less important than completion
    weights = mask.float().masked_fill(mask == -1, cfg.prompt_loss_weight).masked_fill(mask == 1, cfg.completion_loss_weight)
    loss *= weights

    loss = torch.mean(loss)

    return loss

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
    assert len(logits.shape) == 3  # [B, max_seq_len, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_len]
    assert targets.shape == mask.shape  # [B, max_seq_len]
    assert logits.shape[0] == targets.shape[0]

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # only include completion when compute accuracy
    weights = mask.float().masked_fill(mask == -1, 0)

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)

    correct = pred.eq(targets.view_as(pred)).float()

    # only consider completion when compute metrics
    correct *= weights
    num_accurate = correct.sum().item()
    num_samples = weights.bool().sum().item()

    return (num_accurate, num_samples)

def main():
    pass

if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    torch.set_float32_matmul_precision("high")
    
    main()