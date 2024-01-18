"""
Module for build instruct fine-tuning datasets.
"""
import torch

import os
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
from instruct_llama.utils.prompt_builder import build_prompt_completion, Dialog
from instruct_llama.utils.file_helper import read_jsonl_file, count_words

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
        
        assert prompt_tokens is not None and completion_tokens is not None
        
        pairs.append({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})
        
    return pairs

def _raw_texts_to_dialog(dialog_texts: List[str]) -> Dialog:
    """Converts a list of raw text into dialog formation.
    None it requires the texts follows the correct user/assistant/user/assistant ... order.
    """
    
    # requires at least one turn from each role (user, assistant)
    if len(dialog_texts) < 2:
        return None
    
    # try to trim the last one so we always get a pair of contents from user and assistant
    if len(dialog_texts) % 2 != 0:
        dialog_texts = dialog_texts[:-1]
        
    assert len(dialog_texts) % 2 == 0, f"dialog length: {len(dialog_texts)}"
    
    dialog = DEFAULT_DIALOG + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": raw_text.strip()} for i, raw_text in enumerate(dialog_texts)
    ]
    
    return dialog


# ----------------------- High quality datasets ----------------------------------------

def process_dolly_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    max_seq_length: int = 2048, # prompt + completion lengths greater than this are discarded
    overwrite_output: bool = False,
    metadata: Metadata = {
        "name": "Dolly",
        "language": "English",
        'home_page': 'https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm',
    }
) -> None:
    """Process dolly dataset and save the tokenized prompt:completion pairs to .pkl format
    
    Here's an example format of prompt:completion pair before tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": {1st response} </s>" }
    
    """
    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2
    
    train_output_file = os.path.join(output_dir, "train.pkl")
    val_output_file = os.path.join(output_dir, "validation.pkl")
    meta_output_file = os.path.join(output_dir, "meta.json")
    
    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exist, aborting ...'
        )
        return
    
    if metadata is None:
        metadata = {}
        
    json_objs = read_jsonl_file(src_file)
    
    if json_objs is None:
        logger.error(f"Invalid content from src file '{src_file}'")
        
    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)
    
    logger.info("Processing dolly dataset ...")
    datasets = []
    
    for item in json_objs:
        context = item["context"].strip()
        prompt = item["instruction"].strip()
        completion = item["response"].strip()
        
        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue
        
        if len(completion) == 0:
            continue
        
        if len(context) > 0:
            prompt += f"\n\n{context}"
            
        dialog = DEFAULT_DIALOG + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        
        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)
        
        assert prompt_tokens is not None and completion_tokens is not None
        
        if len(prompt_tokens) + len(completion_tokens) > max_seq_length:
            continue
        
        datasets.append({"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})
        
    metadata["vocab_size"] = tokenizer.vocab_size
    metadata["data_structure"] = "A list of prompt:completion token sequences pairs."
    
    logger.info("Saving processed dolly dataset ...")
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata
    )
    
def process_alpaca_dataset(
    src_file: str,
    output_dir: str,
    tokenizer: Tokenizer,
    min_prompt_words: int = 5,
    validation_ratio: float = 0.05,
    overwrite_output: bool = False,
    metadata: Metadata = {
        "name": "Alpaca_cleaned",
        "language": "English",
        'home_page': 'https://github.com/gururise/AlpacaDataCleaned',
    }
) -> None:
    """Process alpaca dataset and save the tokenized prompt:completion pairs to .pkl format.
    
    Here's an example format of prompt:completion pair before apply tokenization:
    {"prompt": "<s>[INST] <<SYS>>\n{system prompt}\n<</SYS>>\n\n{1st user prompt} [/INST]", "completion": " {1st response} </s>"}
    
    """
    
    assert os.path.exists(src_file) and os.path.isfile(src_file)
    assert 0 <= validation_ratio <= 0.2

    train_output_file = os.path.join(output_dir, 'train.pkl')
    val_output_file = os.path.join(output_dir, 'validation.pkl')
    meta_output_file = os.path.join(output_dir, 'meta.json')

    if any(os.path.exists(f) for f in (train_output_file, val_output_file, meta_output_file)) and not overwrite_output:
        logger.error(
            f'The output files "{train_output_file}", "{val_output_file}", "{meta_output_file}" already exists, aborting ...'
        )
        return

    if metadata is None:
        metadata = {}

    json_objs = read_json_file(src_file)

    if json_objs is None:
        logger.error(f'Invalid content from src file "{src_file}"')
        return

    # Create the output directory if necessary
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    logger.info('Processing alpaca dataset ...')
    datasets = []

    for item in json_objs:
        context = item['input'].strip()
        prompt = item['instruction'].strip()
        completion = item['output'].strip()

        # handle special cases where all prompt words are mixed together
        if count_words(prompt) < min_prompt_words:
            continue

        if len(completion) == 0:
            continue

        if len(context) > 0:
            prompt += f'\n\n{context}'

        dialog = DEFAULT_DIALOG + [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': completion},
        ]

        prompt_tokens, completion_tokens = build_prompt_completion(dialog, tokenizer)

        assert prompt_tokens is not None and completion_tokens is not None

        datasets.append({'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens})

    metadata['vocab_size'] = tokenizer.vocab_size
    metadata['data_structure'] = 'A list of prompt:completion token sequences pairs.'

    logger.info('Saving processed alpaca dataset ...')
    _split_and_save_datasets(
        datasets,
        validation_ratio,
        train_output_file,
        val_output_file,
        meta_output_file,
        metadata,
    )
