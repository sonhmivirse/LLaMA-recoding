"""Run supervised fine-tuning (SFT) using QLoRA, starting with pretrained model."""

import torch
import torch.nn.functional as F

# support running without installing as a package
from pathlib import Path
import sys
import numpy as np
import random
import os
import functools
from typing import Tuple

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.utils.logging import create_logger
from instruct_llama.configs.sft_lora import config as cfg
from instruct_llama.tokenizer import Tokenizer
from instruct_llama.utils.custom_dataset import FineTuneDataset

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

def custom_collate_fn(batch, pad_id: int, max_seq_len: int, full_pad: bool = False) -> Tuple[torch.Tensor]:
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    and compute the loss mask for the batch.
    """
    
    batch_size = len(batch)
    
    max_batch_seq_len = max([len(item[0]) + len(item[1]) for item in batch])
    assert max_batch_seq_len <= max_seq_len
    
    if full_pad:
        max_batch_seq_len = max_seq_len
        
    # Concatenate prompt, completion together
    batch_sequences = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)
    
    # loss mask where -1s are prompt tokens, 1s are completion tokens, and 0s are padding tokens
    loss_mask = torch.full((batch_size, max_batch_seq_len), 0, dtype=torch.long)
    
    for i, (prompt, completion) in enumerate(batch):
        # need prompt, completion lengths to compute loss mask
        prompt_len, completion_len = len(prompt), len(completion)
        seq_len = prompt_len + completion_len
        seq = torch.concat((prompt, completion), dim=0).type(torch.long)
        
        # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
        batch_sequences[i, :seq_len] = seq
        loss_mask[i, :prompt_len] = -1 # prompt tokens
        loss_mask[i, prompt_len : prompt_len + completion_len] = 1 # completion tokens
        
    x = batch_sequences[:, :-1] # [batch_size, max_batch_seq_len - 1]
    y = batch_sequences[:, 1:] # [batch_size, max_batch_seq_len - 1]
    
    # shift to right to align with y
    loss_mask = loss_mask[:, 1:]
    
    return x, y, loss_mask


    

def main():
    assert cfg.num_epochs >= 1
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 1
    assert cfg.val_interval >= 1
    assert cfg.val_iters >= 1

    batch_size = int(cfg.micro_batch_size * cfg.gradient_accum_steps)
    
    assert batch_size >= 1

    if not torch.version.cuda:
        raise RuntimeError('This script requires Pytorch with CUDA.')

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint {cfg.pretrain_ckpt_file!r}, aborting ...')
    
    # -------------------- Load datasets -------------------------
    
    logger.info("Loading datasets ...")
    
    tokenizer = Tokenizer(cfg.tokenizer_file)
    
    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=cfg.max_seq_len,
        full_pad=cfg.full_pad
    )
    
    cuda_kwargs = {
        "collate_fn": _collate_fn,
        "num_workers": cfg.dataloader_workers,
        "pin_memory": True,
        "shuffle": False
    }
    
    train_dataset = FineTuneDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)
    

if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    torch.set_float32_matmul_precision("high")
    
    main()