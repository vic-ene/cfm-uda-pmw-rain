
import os
import sys 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print(matplotlib.get_backend())


import shutil
import glob
from natsort import natsorted
import sys


import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

from lightning.fabric import Fabric, seed_everything

import torchmetrics

from omegaconf import DictConfig, OmegaConf
import hydra

import time

import tqdm

from lightning.fabric.utilities.data import AttributeDict


from utils.args_utils import *
from utils.lazy_utils import * 
from utils.ds_utils import *

from fit_model import FitModel





def train(cfg):
    # Create fabric trainer
    # ------------------------------------------------------------------------------------------------------------------------
    seed_everything(cfg["seed"])


    output_dir = cfg["output_dir"]
    print("This is the output dir", output_dir)

    logger_name = "lightning_logs"
    logger_version = "version_0"
    logger = L.pytorch.loggers.TensorBoardLogger(save_dir=output_dir, name=logger_name, version=logger_version)
    logger_dir = os.path.join(output_dir, "", logger_name, "", logger_version)
    ckpt_dir = os.path.join(logger_dir, "", "checkpoints")

    
    fabric = Fabric(  
        accelerator = "gpu",
        num_nodes = cfg["num_nodes"],  
        devices = cfg["num_devices"],
        strategy = "ddp",
        precision = cfg["trainer_precision"], 
        loggers = logger,
    )
    fabric.launch()
    device = fabric.device
    torch.manual_seed(cfg["seed"])

    if fabric.global_rank == 0: os.makedirs(ckpt_dir, exist_ok=True)
    if fabric.global_rank == 0: os.makedirs(logger_dir, exist_ok=True)


    add_variable_to_hydra_cfg(cfg, "logger_dir", logger_dir)
    add_variable_to_hydra_cfg(cfg, "ckpt_dir", ckpt_dir)

    ch = cfg["dataset"]["ch"]
    h = cfg["dataset"]["h"]
    w = cfg["dataset"]["w"]
    dim = ch * h * w

    add_variable_to_hydra_cfg(cfg, "dim", dim)
    add_variable_to_hydra_cfg(cfg, "ch", ch)
    add_variable_to_hydra_cfg(cfg, "h", w)
    add_variable_to_hydra_cfg(cfg, "w", w)

    nfe = cfg["nfe"]
    train_size = cfg["dataset"]["train_size"]
    num_classes = cfg["dataset"]["num_classes"]
    world_size = fabric.world_size

    if cfg["debug"]:
        cfg["dataset"]["ipc"] = 64
        cfg["torch_compile"] = 0



    # --------------------------------------------------------------------------------------------------------------------------------------------

    datamodule = DataModule(cfg, fabric)
    dl_train, sampler_train, dl_test = datamodule.setup()
    dl_train = fabric.setup_dataloaders(dl_train, use_distributed_sampler=False)


    ds_name = cfg["dataset"]["name"]
    with fabric.init_module():
        seed_everything(cfg["seed"])
        
        model_used = cfg["model_used"]
        cfm_used = cfg["cfm_used"]


        unet_cfg = {
            "dim": (ch, h, w), 
            "num_channels": cfg["num_channels"], 
            "num_res_blocks": cfg["num_res_blocks"],
            "attention_resolutions": cfg["attention_resolutions"],
            "channel_mult": cfg["channel_mult"],
        } 

        model_fit = FitModel(
            unet_cfg, cfg=cfg, num_classes=num_classes, device=device,
        )
        

    n_epochs = cfg["n_epochs"]
    model_fit.fit(
        device=device, 
        dl_train=dl_train, sampler_train=sampler_train, dl_test=dl_test,
        n_epochs=n_epochs, logger=logger, 
        fabric=fabric,
    )


      

@hydra.main(version_base=None, config_path="___configs/", config_name="main_config")
def my_app(cfg):
    HPC_PC = is_HPC_PC()
    if HPC_PC:
        print("running on HPC PC")
        import idr_torch

    # Load the configs and all ... 
    # ------------------------------------------------------------------------------------------------------------------------
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #print(hydra_cfg)

    output_dir = hydra_cfg['runtime']['output_dir']
    add_variable_to_hydra_cfg(cfg, "output_dir", output_dir)
    print("This is the output dir", output_dir)
    output_dir_hydra = os.path.join(output_dir, "", ".hydra")
    print(output_dir_hydra)
    custom_save_cfg_file = os.path.join(output_dir_hydra, "", "cfg.yaml")
    print(cfg)

    config_name_retrieved = hydra_cfg['job']['config_name']
    config_dir_retrieved = hydra_cfg['runtime']['config_sources'][1]["path"]
    config_path_retrieved = os.path.join(config_dir_retrieved, "", config_name_retrieved + ".yaml")


    if os.environ.get('SLURM_NNODES'):
        cfg["num_nodes"] = int(os.environ['SLURM_NNODES'])
        print("num nodes here",  cfg["num_nodes"])
    if os.environ.get('SLURM_GPUS_ON_NODE'):
        cfg["num_devices"] = int(os.environ['SLURM_GPUS_ON_NODE'])
        print("num gpus here", cfg["num_devices"])
    if os.environ.get('SLURM_JOB_GPUS'):
         extra_var = os.environ.get('SLURM_JOB_GPUS')
         print("This is extra variables", extra_var)

    cfg["num_gpus"] = cfg["num_nodes"] * cfg["num_devices"]
    print("Counted number of gpus: ", cfg["num_gpus"])

    deterministic = cfg["deterministic"]
    torch_compile = cfg["torch_compile"]
    if torch_compile:
        deterministic = False


    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)     
        torch.backends.cudnn.benchmark = True         
        torch.backends.cudnn.deterministic = False    
    torch.set_float32_matmul_precision('high')
    

    # Copy the original config file in hydra dir
    shutil.copy(config_path_retrieved, custom_save_cfg_file)
    add_variable_to_hydra_cfg(cfg, "cfg_full_path", custom_save_cfg_file)
    # print("This was the config path retrieved", config_path_retrieved)
    # print("this is the cfg", cfg)

    train(cfg)

if __name__ == "__main__":
    my_app()