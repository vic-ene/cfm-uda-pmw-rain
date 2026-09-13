import sys 
import os
import copy
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict, DictConfig
import hydra

from pathlib import Path


def initialize_vars(instance, variables, exclude=[]):
    """
    https://stackoverflow.com/questions/12191075/is-there-a-shortcut-for-self-somevariable-somevariable-in-a-python-class-con
    Initialize instance variables with the given dictionary of variables,
    while excluding variables listed in 'exclude'.
    """
    
    # Check for existing variable conflicts
    if any(k in vars(instance) for k in variables):
        raise Exception("Cannot allow overriding existing variable, check to see what to do")
    
    # Update instance variables excluding those in the 'exclude' list
    vars(instance).update((k, v) for k, v in variables.items() if k != 'self' and k not in exclude)





def pickle_save(file, item):
    import pickle
    import zstandard as zstd

    cctx = zstd.ZstdCompressor(level=10)
    with open(file, 'wb') as fp:
        fp.write(cctx.compress(pickle.dumps(item)))

def pickle_load(file):
    import pickle
    import zstandard as zstd
    from filelock import FileLock

    ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'  # zstd magic bytes

    with FileLock(f'{file}.lck'):
        with open(file, 'rb') as pfile:
            raw = pfile.read()

    if raw[:4] == ZSTD_MAGIC:
        # new format: zstd compressed
        dctx = zstd.ZstdDecompressor()
        data = pickle.loads(dctx.decompress(raw))
    else:
        # legacy format: plain pickle
        data = pickle.loads(raw)

    return data


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def count_params_millions(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    to_m = lambda x: x / 1e6  # convert to millions

    print(f"Total parameters:       {to_m(total):.3f} M")
    print(f"Trainable parameters:   {to_m(trainable):.3f} M")
    print(f"Non-trainable params:   {to_m(non_trainable):.3f} M")

    return total




def add_variable_to_hydra_cfg(cfg, var, val):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        OmegaConf.update(cfg, var, val)
    OmegaConf.set_struct(cfg, False)




def is_HPC_PC():
    HPC_PC = True
    if "vicene" in os.getcwd() \
    or os.getcwd().startswith("/home/enescu") \
    or os.getcwd().startswith("/home/venescu") \
    or os.getcwd().startswith("/net/nfs/ssd3"):
        HPC_PC = False
    return HPC_PC


