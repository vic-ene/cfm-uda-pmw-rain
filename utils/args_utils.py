from omegaconf import OmegaConf, open_dict, DictConfig
import os

def add_variable_to_hydra_cfg(cfg, var, val):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        OmegaConf.update(cfg, var, val)
    OmegaConf.set_struct(cfg, False)



