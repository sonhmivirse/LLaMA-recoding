
from pathlib import Path
import logging

import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp.api import (
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig
)

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

logger = logging.getLogger(__name__)


full_state_model_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
full_state_optim_config = FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=True)

def save_full_state_model_checkpoint(model, rank, ckpt_save_path, overwrite=False):
    """saving modle via rank0 cpu streaming and full_state_dict"""
    
    save_full_path = Path(ckpt_save_path)
    if rank == 0:
        if not overwrite and save_full_path.exists():
            logger.warning(f"a file with the same name already exists at {save_full_path}, aborting ...")
            return
        else:
            save_dir = save_full_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
    
    with FSDP.state_dict_type(
        model,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=full_state_model_config,
        optim_state_dict_config=full_state_optim_config
    ):
        model_state = model.state_dict()
        
    logger.debug(f"model state_dict ready on rank {rank}\n")
    
    if rank == 0:
        logger.debug('--> saving model ...')
        
        torch.save(model_state, save_full_path)
        
        logger.debug(f"--> model checkpoint saved at {save_full_path}\n")
        
        