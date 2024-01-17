"""
Module for build instruct fine-tuning datasets.
"""
import torch

from typing import Text, Mapping, List
import random
import pickle
import json

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from instruct_llama.utils.logging import create_logger
from instruct_llama.tokenizer import Tokenizer

logger = create_logger()

Metadata = Mapping[Text, Text]

DEFAULT_SYSTEM_PROMPT = {
    "role": "system",
    "content": ""
}

# this will be inserted into the training data as the first system prompt
DEFAULT_DIALOG = [DEFAULT_SYSTEM_PROMPT]

# ------------------------------------- helper functions --------------------------------------------------

def _split_and_save_datasets(
    datasets: List[dict],
    validation_ratio: float,
    train_output_file: str,
    val_output_file: str,
    meta_output_file: str,
    meta: dict
) -> None:
    # split into train and validation datasets as dolly only have one single .json file
    random.shuffle(datasets)
    
    val_size = int(len(datasets) * validation_ratio)
    train_size = len(datasets) - val_size
    
    train_set, val_set = torch.utils.data.random_split(datasets, [train_size, val_size])
    
    for data, out_file in zip((train_set, val_set), (train_output_file, val_output_file)):
        if len(data) > 0:
            logger.info(f"Saving {len(data)} processed data to {out_file!r} ...")
            pickle.dump(data, open(out_file, "wb"))
            
    meta = {
        **meta,
        "num_train_samples": len(train_set),
        "num_validation_samples": len(val_set)
    }
    
    logger.info(f"Saving metadata to {meta_output_file!r} ...")
    
    with open(meta_output_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        

def _process_single_dm_math_txt_file(txt_file: str, tokenizer: Tokenizer) -> List[dict]:
    pairs = []
    
    with open(str(txt_file), "r", encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]
        
    for i in range(0, len(lines), 2):
        prompt = lines[i].strip()
        completion = lines[i+1].strip()
        
        dialog = DEFAULT_DIALOG + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        
        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)