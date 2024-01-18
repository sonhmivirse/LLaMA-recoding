import os
from typing import Iterable, Text, Any, Mapping
import gzip
import json


def read_jsonl_file(input_file: str) -> Iterable[Mapping[Text, Any]]:
    """Generator yields a list of json objects or None if input file not exists or is not .jsonl file."""
    if not os.path.exists(input_file) or not os.path.isfile(input_file) or not input_file.endswith(".jsonl"):
        return None
    
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()
        return json.loads(content)

def read_zipped_jsonl_file(input_file: str) -> Iterable[Mapping[Text, Any]]:
    """Generator yields a list json objects or None if input file not exists or is not .jsonl file."""
    if(
        not os.path.exists(input_file)
        or not os.path.isfile(input_file)
        or (not input_file.endswith(".json.gz") and not input_file.endswith(".jsonl.gz"))
    ):
        return None
    
    with gzip.open(input_file, "rt", encoding="utf-8") as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.decoder.JSONDecodeError as e:
                print(f"{e}, skip line in file {input_file}")
                continue

def count_words(raw_text, is_chinese=False) -> int:
    if raw_text is None or raw_text == "":
        return 0
    
    if is_chinese:
        return len(raw_text)
    
    return len(raw_text.split())

def find_certain_files_under_dir(root_dir: str, file_type: str = ".txt") -> Iterable[str]:
    """Given a root folder, find all files in this folder and it's sub folders that matching the given file type."""
    assert file_type in [".txt", ".json", ".jsonl", ".parquet", ".zst", ".json.gz", ".jsonl.gz"]
    
    files = []
    if os.path.exists(root_dir):
        for root, dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(file_type):
                    files.append(os.path.join(root, f))
                    
    return files

def count_words(raw_text, is_chinese=False) -> int:
    if raw_text is None or raw_text == "":
        return 0
    
    if is_chinese:
        return len(raw_text)
    
    return len(raw_text.split())

