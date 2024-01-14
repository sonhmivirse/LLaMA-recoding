
from typing import Iterable, List
import os
import json
import numpy as np
import random
import math
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch

class DataSource:
    """A simple class to load data source from numpy.memmap structure."""
    
    def __init__(self, name, weights, data_file, metadata_file):
        self.name = name
        self.weights = weights
        self.data_file = data_file
        self.metadata_file = metadata_file
        
        self.num_tokens = 0
        self.data_type = None
        self.data = None
        self.metadata = None
        
        self.sanity_check()
    
    def _sanity_check(self):
        assert len(self.name) > 0
        assert 0 <= self.weights <=1
        assert os.path.exists(self.data_file) and self.data_file.endswith('.npy')
        assert os.path.exists(self.metadata_file) and self.metadata_file.enswith(".json")
        
    def update_weights(self, v) -> None:
        assert 0 <= v <=1
        self.weights = v
        
    def load_metadata(self) -> None:
        with open(self.metadata_file, 'r') as file:
            metadata = json.load(file)
            
        assert "num_tokens" in metadata and "data_type" in metadata
        assert metadata["data_type"] in ["uint16", "uint32"]
        
        self.weights = self.weights if self.num_tokens > 0 else 0.0
        
    def load_data(self, fully_load: bool = False) -> None:
        if self.weights > 0 and self.num_tokens > 0:
            arr_memmap = np.memmap(self.data_file, dtype=self.data_type, mode='r', shape=(self.num_tokens,))
            
            # load the entire dataset into memory
            if fully_load:
                self.data = np.array(arr_memmap)
                del arr_memmap
            else:
                self.data = arr_memmap
    
    def extract_metadata(self):
        return {
            "name": self.name,
            "weights": self.weights,
            "num_tokens": self.num_tokens,
            "vocab_size": self.metadata["vocab_size"],
            "data_type": self.metadata["data_type"],
            "data_file": self.data_file,
            "metadata_file": self.metadata_file
        }
        

class BlendedDataset(IterableDataset):
    """A blended dataset used for pre-training.
    
    It supports mixed data sources, where we can define the weights of each data source.
    Additionally, it also supports shard the dataset based on the world size.
    """
    
    def __init__(
        self,
        data_sources: Iterable[DataSource],
        max_seq_len: int,
        rank=0,
        world_size=0,
        seed: int = 1,
        fully_load: bool = False,
    ) -> None:
        """
        Args:
            data_sources: a list of DataSource objects to specify where to load the data, and how often we should use it
            max_seq_len (int): the context window (or block size) of the GPT model.
            rank (int): rank of the process to shard data, default to 0
            world_size (int, optional): how many partitions to use when shard data. Defaults to 0 no shard.
            seed (int, optional): random seed. Defaults to 1.
            fully_load (bool, optional): load the entire dataset into memory. default off.
        """
        
        assert len(data_sources) > 0
        assert max_seq_len is not None and max_seq_len > 0
        assert rank is not None and rank >= 0
        assert world_size is not None and world_size >= 0
        
        random.seed(seed)
        
        self.data_sources: Iterable[DataSource] = data_sources
        
        self.rank = rank
        self.world_size = world_size
        
        self.max_seq_len = max_seq_len
        self.fully_load = fully_load
        self.total_num_tokens = 0
        
        
        # Load the data sources
        for source in self.data_sources:
            source.load_metadata()
            source.load_data(self.fully_load)
            self.total_num_tokens += source.num_tokens
            
        assert self.total_num_tokens > 0
        
        # Derive and normalize data source sampling probabilities
        sample_weights = np.array([source.weights for source in self.data_sources], dtype=np.float16)
        assert 0 < np.sum(sample_weights)
        
        self.sample_weights = (sample_weights / np.sum(sample_weights)).tolist()
        
        for source, p in zip(self.data_sources, self.sample_weights):
            source.update_weights(p)
            
        # pre-compute shard start and end indices for each data source
        self.shard_indices = []
        
        for source in self.data_sources:
            num_tokens = source.num_tokens
            start_idx = 0
            end_idx = num_tokens - 1
            
            if self.world_size > 1:
                shard_size = int(math.ceil(num_tokens / float(self.world_size)))
                
                start_idx = shard_size * self.rank
                end_idx = start_idx + shard_size
                
                if end_idx > num_tokens - 1:
                    end_idx = num_tokens - 1
                    
                assert start_idx >= 0 and end_idx - start_idx > self.max_seq_len
                
            start_indices = [
                i
                for i in range(0, end_idx - self.max_seq_len, self.max_seq_len)
                if i * self.max_seq_len + max_seq_len < end_idx
            ]
            self.shard_indices.append(start_indices)
            
    def generator(self):
        while True:
            ds_idx = self._choose_datasource()
            source = self.data_sources[ds_idx]
            
            num_tokens = source.num_tokens
            data = source.data
            
            assert data is not None and num_tokens > self.max_seq_len
            
            # Get shard start indices for the chosen data source
            start = random.choice(self.shard_indices[ds_idx])
            end = start + self.max_seq_len
            
            assert end <= num_tokens - 1
            
            # here the high is exclusive
            x = torch.from_numpy((data[start:end]).astype(np.int32)).to(dtype=torch.long)
            y = torch.from_numpy((data[start + 1: end + 1]).astype(np.int32).to(dtype=torch.long))
            
            yield x, y
            
    def __iter__(self):
        return iter(self.generator())
    
    def _chose_datasource(self) -> int:
        return random.choices(range(len(self.data_sources)), weights=self.sample_weights, k=1)[0]
    
    def __len__(self):
        return int(self.total_num_tokens / self.max_seq_len)
    
    def get_metadata(self):
        return {
            "dataset_type": "blended",
            "num_tokens": self.total_num_tokens,
            "fully_loaded": self.fully_load,
            "data_sources": [source.extract_metadata() for source in self.data_sources]
        }