from pathlib import Path
import sys
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from instruct_llama.utils.custom_dataset import FineTuneDataset
from instruct_llama.configs.sft_lora import config as cfg

train_dataset = FineTuneDataset(data_sources=cfg.train_datasources, max_seq_len=cfg.max_seq_len)
print(train_dataset[0]) 