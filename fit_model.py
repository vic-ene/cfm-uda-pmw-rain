import os
import shutil
import glob
from natsort import natsorted

import torch
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging

from lightning.fabric import Fabric, seed_everything

from torchdyn.core import NeuralODE

import torchmetrics


from omegaconf import DictConfig, OmegaConf
import hydra

import time

from tqdm import tqdm

from lightning.fabric.utilities.data import AttributeDict

from utils.args_utils import *
from utils.ds_utils import *
from utils.plot_utils import *
from utils.my_nn_utils import *
from utils.my_tensor_utils import *
from utils.satellite_data_utils import * 
from utils.colloc_utils import *


import torchdiffeq
from torchcfm.models.unet.unet import UNetModel
from torchcfm.conditional_flow_matching import *
from model_dit.model_dit import DiT
from model_dit.dit_no_t_emb import DiT as DiT_no_t_emb
from utils.merged_cfm import MergedCFM, ODE_FULL_ARR, ODE_INVERSION_ARR

from pathlib import Path

from collections import defaultdict
import copy


def count_parameters_and_buffers(model, msg=""):
    print("---")
    # Compute total parameters, gradient-bearing parameters, and buffers
    total_params = sum(p.numel() for p in model.parameters())
    total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model.buffers())

    print(f"Total parameters: {total_params}"
          f"Total parameters with grad: {total_params_grad}"
          f"Total buffers: {total_buffers}",
          msg)

    total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    total_buffers = sum(b.numel() for b in model.buffers()) / 1_000_000

    print(f"Total parameters: {total_params:.2f}M, " 
          f"Total parameters with grad: {total_params_grad:.2f}M, "
          f"Total buffers: {total_buffers:.2f}M",
          msg)
    print("---")

    return total_params



@torch.no_grad()
def ema_fn(source, target, decay=0.9999):
    for p_tgt, p_src in zip(target.parameters(), source.parameters()):
        # Equivalent to: p_tgt = p_tgt * decay + p_src * (1 - decay)
        p_tgt.mul_(decay).add_(p_src, alpha=1 - decay)


class FitModel(torch.nn.Module):
    def __init__(self, unet_cfg, cfg, num_classes, device, dtype=torch.float32) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg_ds = self.cfg["dataset"]
        self.ds_name = self.cfg_ds["name"]
        self.unet_cfg = unet_cfg

        self.num_classes = num_classes

        self.dtype = dtype
        self.device = device

        self.ch = self.cfg["ch"]
        self.h = self.cfg["h"]
        self.w = self.cfg["w"]
        self.dim = self.cfg["dim"]

        self.nfe = self.cfg["nfe"]
        self.solver = self.cfg["solver"]
        self.cfm_used = cfg["cfm_used"]

        self.ds_name = cfg["dataset"]["name"]

        self.class_dropout_prob = cfg["class_dropout_prob"]
        self.cond_used = cfg["cond_used"]

        self.noise_type=self.cfg["noise_type"]

        self.model = self.get_model(
            self.unet_cfg, 
            self.num_classes,
        )


        self.FM = MergedCFM(
            time_sampler=cfg["time_sampler"],
            noise_type=cfg["noise_type"],
            cfm_used=cfg["cfm_used"],
            logit_normal_mu_t=cfg["logit_normal_mu_t"],
            logit_normal_std_t=cfg["logit_normal_std_t"],
            si_gamma_fn=cfg["si_gamma_fn"],
            si_gamma_a=cfg["si_gamma_a"],
            cfg=cfg,
            device=device,
        )




        print(count_parameters_and_buffers(self.model, msg="count_params"))
        self.ema = None
        self.ema_epoch_start = self.cfg["ema_epoch_start"]
        self.ema_decay = self.cfg["ema_decay"]


        self.logit_normal_mu = self.cfg["logit_normal_mu"]
        self.logit_normal_sigma = self.cfg["logit_normal_sigma"]
        print("this is logitnormal mu and sigma", self.logit_normal_mu, self.logit_normal_sigma)


    def get_model(self, unet_cfg, num_classes):
        cfg = self.cfg
        self.model_used = self.cfg["model_used"]


        # num_classes=1000
        if self.model_used == "unet":
            h = cfg["h"]
            in_channels = cfg["ch"] 
            class_dropout_prob = cfg["class_dropout_prob"]

            model = UNetModel(
                image_size=h,
                in_channels=in_channels,
                out_channels=in_channels,
                model_channels=cfg["num_channels"],
                num_res_blocks=cfg["num_res_blocks"],
                attention_resolutions=cfg["attention_resolutions"],
                channel_mult=cfg["channel_mult"],
                num_classes=num_classes,
                class_dropout_prob=class_dropout_prob,
                cfg=cfg,
            )


        elif self.model_used in  "dit":
            _it_cfg = self.cfg["_it"]
            depth = _it_cfg["depth"]
            hidden_size = _it_cfg["hidden_size"]
            patch_size = _it_cfg["patch_size"]
            num_heads = _it_cfg["num_heads"]
            class_dropout_prob = cfg["class_dropout_prob"]

            print("info from dit model", _it_cfg, self.ch, self.h)

            model = DiT(
                depth=depth, hidden_size=hidden_size, patch_size=patch_size, num_heads=num_heads, 
                in_channels=self.ch, input_size=self.h, 
                num_classes=num_classes, class_dropout_prob=class_dropout_prob,
                learn_sigma=False,
                cfg=self.cfg,
            )

       
        else:
            sys.exit(" do not recognize model")



        return model 
        
            
    def warmup_lr(self, step):
        return min(step, self.cfg["warmup_steps"]) / self.cfg["warmup_steps"]


    def ode_field(self, model_, t, x, y, cfg_scale):
        if self.cond_used == "cfg":
            y_cond = y
            y_uncond = torch.ones_like(y) * (model_.y_embedder.cfg_label)

            vel_cond = model_.forward(t, x, y_cond)
            vel_uncond = model_.forward(t, x, y_uncond)
            vel = cfg_scale * vel_cond + (1 - cfg_scale) * vel_uncond
            return vel
            
        elif self.cond_used == "vanilla":
            return model_.forward(t, x, y)
        else:
            sys.exit("do not recognise conditioning for sampling")


    def generate_samples(self, model, noise, nfe, device, y=None, method="neural_ode", t_start=0, t_end=1, cfg_scale=1.0):
        model.eval()
        model_ = copy.deepcopy(model)
        if "fabric" in str(type(model)): model_ = model_.module.to(device)


        with torch.no_grad():
            traj = torchdiffeq.odeint(
                lambda t, x: self.ode_field(model_, t, x, y, cfg_scale),
                noise,
                torch.linspace(t_start, t_end, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method=method,
                options=dict(step_size=1/nfe),
            )[-1]

        return traj
        

    

    def encode_decode_samples(self, model, FM, x, y1, y2, t_enc=0.5, num_steps=1, method="dopri5", atol=1e-4, rtol=1e-4,
            cfg_scale_forward=1.0, cfg_scale_reverse=1.0, 
            do_stochastic_noising=0, cfm_used_override=None, 
            cut_into_patches=True, overlap_size=1, overlap_mode="mean"):
            

        size = getattr(self.model, "input_size", None) or getattr(self.model, "image_size", None)
        x_og = x.clone()
        y1_og = y1.clone()
        y2_og = y2.clone()

        if cut_into_patches:
            x, x_meta = cut_tensor_into_square_overlapping_patches(x, size, overlap_size)
            dupl_factor = x.shape[0] // x_og.shape[0]
            y1 = y1.repeat(dupl_factor)
            y2 = y2.repeat(dupl_factor)


        x_rec = FM.adapt_torchdiffeq(
            model, x, y1, y2,
            t_enc=t_enc, num_steps=num_steps, method=method, atol=atol, rtol=rtol,
            cfg_scale_forward=cfg_scale_forward, cfg_scale_reverse=cfg_scale_reverse,
            do_stochastic_noising=do_stochastic_noising,
            cfm_used_override=cfm_used_override,
        )
       
        if cut_into_patches:
            x_rec = stitch_square_overlapping_patches_back(x_rec, x_meta, overlap_mode)

        return x_rec.cpu()



    def init_ema(self):
        self.ema = copy.deepcopy(self.model)
        self.ema.eval()


    def fit(self, device=None, dl_train=None, sampler_train=None, dl_test=None, n_epochs=200, logger=None, fabric=None, dtype=torch.float32) -> None:    
        cfg = self.cfg

        lr_used = self.cfg["lr"]
        param_groups = self.model.parameters()

        self.optimizer = torch.optim.Adam(param_groups, lr=lr_used)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.warmup_lr)
        if self.cfg["torch_compile"]:
            print("compiling model that is trained")
            self.model = torch.compile(self.model)

        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

        self.dtype = dtype

        print("this is the sampler_train", sampler_train)

        debug = cfg["debug"]
        

    
        if not cfg["skip_to_test"]:
            for epoch_idx in range(n_epochs):
                sampler_train.set_epoch(epoch_idx)
                print("this is the epoch_idx", epoch_idx)
                if ((epoch_idx + 1) == self.ema_epoch_start) or (self.ema_epoch_start == 0):
                    # print("initted the ema", epoch_idx) 
                    self.init_ema()

                self.model.train() # Reenable train to be sure (could became unactivated when sampling ... )
                avg_loss = AverageMeter()

                dl_enumerate = tqdm.tqdm(enumerate(dl_train))
                for batch_idx, data in dl_enumerate:
                    if debug and (batch_idx > 3):
                        break

                    x = data[0].to(device)
                    y = data[1].to(device)

                    loss_dict = self.FM.get_loss(self.model, x, y, epoch=epoch_idx)
                    loss = loss_dict["loss"]
                    avg_loss.update(loss.detach().cpu(), x.shape[0])

                                            
                    self.optimizer.zero_grad()
                    fabric.backward(loss)
                    self.optimizer.step()
                    self.sched.step()
                    

                    if (epoch_idx + 1) >= self.ema_epoch_start: 
                        ema_fn(self.model, self.ema, decay=self.ema_decay)

            
                step = epoch_idx
                # Log stuff 
                # ----------------------------------------------------------------------------------------------------------------
                if fabric.global_rank == 0:
                    # Log the loss 
                    # ----------------------------------------------------------------------------------------------------------------
                    prefix="111_loss_info"
                    logger.experiment.add_scalars(f"{prefix}_loss",{
                        "loss": avg_loss.avg,
                    }, global_step=step)
                    # ----------------------------------------------------------------------------------------------------------------

                   
                    # Log the learning rate of the optimizer
                    # ----------------------------------------------------------------------------------------------------------------
                    logger.experiment.add_scalars(f"9999_lr_info", {
                        "lr": self.optimizer.param_groups[0]['lr'],
                    }, global_step=step)
                    # ----------------------------------------------------------------------------------------------------------------
                # ----------------------------------------------------------------------------------------------------------------

             
                    
                # Save sampled images 
                # ----------------------------------------------------------------------------------------------------------------

                # Only train a single model
                if self.ds_name in ["sat_gpm", "sat_f18"]:
                    continue

                LABEL_GPM = cfg["dataset"]["class_to_idx"]["gpm"]
                LABEL_F18 = cfg["dataset"]["class_to_idx"]["f18"]

                S = cfg["dataset"]["h"]
                nfe = cfg["nfe"]

                if ((epoch_idx + 1) % self.cfg["save_every_k_epochs"] == 0) or (epoch_idx == 0):
                    self.model.eval()
                    with torch.no_grad():
                        model_for_sampling = self.model if self.ema == None else self.ema
                        model_for_sampling.eval()

                        NFE = self.cfg["nfe"] if not self.cfg["debug"] else 3
                        self.solver = "euler" if self.cfg["debug"] else self.solver

                        ds_test = dl_test.dataset
                        x_test, y_test = ds_test.data, ds_test.targets
                        x_test = [
                            dl_test.dataset.get_normalized_data_and_labels(device, x=elt_x, y=elt_y)[0]
                            for i, (elt_x, elt_y) in enumerate(zip(x_test, y_test))
                        ]


                        for i, (elt_x, elt_y) in enumerate(zip(x_test, y_test)):
                            if elt_y == LABEL_GPM:
                                continue

                            if elt_y == LABEL_F18:
                                C_LABEL = LABEL_F18
                            else:
                                raise ValueError("Did not plan for this label")

                            t_start = 1.0
                            t_end = 0.5
                            if len(elt_x.shape) == 3:
                                elt_x = elt_x.unsqueeze(0)
                            elt_x = elt_x.to(device)
                            elt_y = elt_y.to(device)


                            x_f18_to_gpm, = self.encode_decode_samples(
                                model=model_for_sampling, FM=self.FM, x=elt_x, y1=elt_y, y2=torch.full_like(elt_y, LABEL_GPM), t_enc=t_end,
                                num_steps=nfe, method=self.solver, atol=1e-4, rtol=1e-4,
                                cfg_scale_forward=1.0, cfg_scale_reverse=1.0,
                                cut_into_patches=True,
                            ).unsqueeze(0)

                            x_denorm = dl_test.dataset.get_unormalized_data_and_labels(device, x_f18_to_gpm, torch.tensor(LABEL_GPM).to(torch.int64))[0]
                            x_denorm_og = dl_test.dataset.get_unormalized_data_and_labels(device, elt_x, torch.tensor(C_LABEL).to(torch.int64))[0]


                            x_show = torch.cat([x_denorm_og, x_denorm])
                            x_show = flatten_channels_and_stack(x_show, stack_vertically=True)
                            save_to_tensorboard_sat(fabric.global_rank, logger, file_grp=f"1114_adapted_{str(elt_y.item())}_{i}", step=step, img=x_show, title=None)

                          


                        # Sample class conditional data from same noise, to validate the method
                        # -------------------------------------------------------------------------------
                        data_arr = []
                        y_arr = []
                        samples_per_class = 2  # was hardcoded as z = torch.randn(2, ...)
                        for class_idx in range(min(self.num_classes, len(cfg["dataset"]["class_to_idx"]))):
                            L.fabric.seed_everything(0)
                            z = torch.randn(samples_per_class, self.ch, self.h, self.w).to(device)
                            y = (torch.ones(z.shape[0]) * class_idx).to(torch.int64).to(z.device)
                            data = self.generate_samples(model=model_for_sampling, noise=z, y=y, nfe=NFE, device=device, method=self.solver)
                            data_arr.append(data)
                            y_arr.append(y)
                        data = torch.cat(data_arr)
                        y = torch.cat(y_arr)
                        data, y = dl_train.dataset.get_unormalized_data_and_labels(device, data, y)
                        # data = x_test_og.clone()

                        # Flatten channels horizontally for each sample -> [h, ch*w]
                        data_arr_bis = []
                        for elt in data:
                            elt = torch.cat([elt_ch for elt_ch in elt], dim=1)
                            data_arr_bis.append(elt)
                        data_arr_bis = torch.stack(data_arr_bis, dim=0)  # [N, h, ch*w]

                        # Build the grid dynamically: one row per class, `cols` images per row
                        N = data_arr_bis.shape[0]
                        cols = samples_per_class
                        rows = []
                        for r in range(0, N, cols):
                            row_imgs = list(data_arr_bis[r:r + cols])
                            while len(row_imgs) < cols:                     # pad incomplete last row
                                row_imgs.append(torch.zeros_like(data_arr_bis[0]))
                            rows.append(torch.cat(row_imgs, dim=1))         # concat along width
                        grid = torch.cat(rows, dim=0)                       # stack rows along height

                        save_to_tensorboard_sat(fabric.global_rank, logger, file_grp=f"1115_sampled_data", step=step, img=grid, title=None)
                        # -------------------------------------------------------------------------------
                        
                self.model.train()
                # ----------------------------------------------------------------------------------------------------------------


            # Save the final checkpoint of the model
            # Before saving, set it to the ema model
            # ----------------------------------------------------------------------------------------------------------------    

            for param in self.model.parameters(): 
                param.grad = None
                    
            if self.ema != None:
                self.model = copy.deepcopy(self.ema)
                for _ in range(3): print("copied ema model")

            sd = AttributeDict(
                model=self.model.state_dict(),
            )
            ckpt_path = os.path.join(self.cfg["ckpt_dir"], f"ckpt_{epoch_idx + 1}.ckpt")
            fabric.save(ckpt_path, sd)
            # ----------------------------------------------------------------------------------------------------------------

        self.test_rain_prediction(fabric, logger, dl_train, dl_test)


    def test_rain_prediction(self, fabric, logger, dl_train, dl_test):
        cfg = self.cfg
        cfg_ds = cfg["dataset"]
        device = fabric.device

        MAX_RAIN_VAL = 1000.0

        
        t_start = 1.0
        rr_threshold=0.1
        
        t_end = 0.5 if not "t_end_override" in cfg else cfg["t_end_override"]
        do_plot = 0 if not "do_plot_override" in cfg else cfg["do_plot_override"]
        overlap_size = 1 if not "overlap_size_override" in cfg else cfg["overlap_size_override"]
        overlap_mode = "mean" if not "overlap_mode_override" in cfg else cfg["overlap_mode_override"]

        do_stochastic_noising = 0 if not "do_stochastic_noising_override" in cfg else cfg["do_stochastic_noising_override"]
        add_variable_to_hydra_cfg(cfg, "do_stochastic_noising", do_stochastic_noising)
        add_variable_to_hydra_cfg(self.cfg, "do_stochastic_noising", do_stochastic_noising)



        nfe = cfg["nfe"] if not "nfe_override" in cfg else cfg["nfe_override"]
        solver = cfg["solver"] if not "solver_override" in cfg else cfg["solver_override"]
        debug = 0 if not "debug_override" in cfg else cfg["debug_override"]        

        print("this is the debug_override", debug)
        print("this is the do_plot", do_plot)



        ckpt_path = cfg["ckpt_path"]
        if ckpt_path != "":
            print("loading ckpt for test", ckpt_path)
            with fabric.init_module():
                seed_everything(cfg["seed"])
                
                model_used = cfg["model_used"]
                cfm_used = cfg["cfm_used"]
                ch = cfg["ch"]
                h = cfg["h"]
                w = cfg["w"]
                num_classes = cfg["dataset"]["num_classes"]

                unet_cfg = {
                    "dim": (ch, h, w), 
                    "num_channels": cfg["num_channels"], 
                    "num_res_blocks": cfg["num_res_blocks"],
                    "attention_resolutions": cfg["attention_resolutions"],
                    "channel_mult": cfg["channel_mult"],
                } 

                ckpt = fabric.load(ckpt_path)
                yaml_root = Path(ckpt_path).parent.parent.parent.parent
                yaml_path = os.path.join(yaml_root, "", ".hydra", "", "config.yaml")
                override_cfg = OmegaConf.load(yaml_path)
         
                self.cfg = OmegaConf.merge(self.cfg, override_cfg)
                cfg = self.cfg
                model_new = self.get_model(unet_cfg, num_classes)
                model_new.cfg = self.cfg

                model_new.load_state_dict(ckpt["model"])
            
                self.model = model_new
                self.model = freeze(self.model)



        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        
        # Load DRAIN model 
        # -------------------------------------------------------------------------------------------  
        ckpt_path_drain = "./huggingface_downloads/ddrain_model/0.0_0.5_1.0_0.0_1/lightning_logs/version_0/checkpoints/ckpt_499.ckpt"
        if not os.path.isfile(ckpt_path_drain):
            raise ValueError("you gotta provide a correct path for the rain model")

        ckpt_drain = fabric.load(ckpt_path_drain)
        config_drain_path = os.path.join(
            Path(ckpt_path_drain).parent.parent.parent.parent, "", ".hydra", "", "config.yaml"
        )
        cfg_drain = OmegaConf.load(config_drain_path)
       

        _it_cfg = cfg_drain["_it"]
        depth = _it_cfg["depth"]
        hidden_size = _it_cfg["hidden_size"]
        patch_size = _it_cfg["patch_size"]
        num_heads = _it_cfg["num_heads"]
        h = 32
        in_channels = cfg_drain["dataset"]["ch"]

        ddrain = DiT_no_t_emb(
            depth=depth, hidden_size=hidden_size, patch_size=patch_size, num_heads=num_heads, 
            in_channels=in_channels, out_channels=1, input_size=h, 
            learn_sigma=False,
            is_emb_active=False,
            cfg=cfg_drain,
            num_classes=0,
        )

        if any(k.startswith("_orig_mod.") for k in ckpt_drain["model"]):
            ckpt_drain["model"] = {k.removeprefix("_orig_mod."): v for k, v in ckpt_drain["model"].items()}
        ddrain.load_state_dict(ckpt_drain["model"])
        ddrain = ddrain.to(device)
        ddrain = freeze(ddrain)
        # ----------------------------------------------------------------------------------------------------

        
        res_dir = os.path.join(os.path.dirname(self.cfg["ckpt_dir"]), "", "res_dir")
        os.makedirs(res_dir, exist_ok=True)


        model_for_sampling = self.model

        cond_used = cfg["cond_used"]
        rr_interval_arr = [
            (0.0, rr_threshold),
            (rr_threshold, 2.5),
            (2.5, 10.0),
            (10.0, 50),
            (50, MAX_RAIN_VAL),
            (0.0, MAX_RAIN_VAL),
        ]



        LABEL_GPM = cfg["dataset"]["class_to_idx"]["gpm"]
        LABEL_F18 = cfg["dataset"]["class_to_idx"]["f18"]
        S = cfg["dataset"]["h"]

                    
        if debug: 
            nfe = 3

        n_pixel_colloc_arr_gt = []
        n_pixel_colloc_arr_gpm = []

        with torch.no_grad():
          

            dict_arr_arr = [
                cfg_ds["path_test_f18"],
            ]


            for dict_path_idx, dict_path in enumerate(dict_arr_arr):
                F18_ROI = 10_000
                F18_NEIGHBORS = 3


                fill_value=float('nan'),
                power=2


                sat_c = None
                if dict_path_idx == 0:
                    sat_c = "f18"
                    C_LABEL = LABEL_F18
                    C_ROI = F18_ROI
                    C_NEIGHBORS = F18_NEIGHBORS


                else:
                    raise ValueError("Did not plan to test for so many different satellites")

                print("this is the currentc_sat_name", sat_c, C_LABEL)
                
                avg_meter_dict_wrt_gt = {}
                for i in range(len(rr_interval_arr)):
                    temp_dict = {
                        "rmse_gpm_pixel": AverageMeter(),
                        "mae_gpm_pixel": AverageMeter(),
                        "bias_gpm_pixel": AverageMeter(),
                        "iou_gpm_pixel": AverageMeter(),
                        "ssim_gpm_pixel": AverageMeter(),
                        
                        f"rmse_{sat_c}_pixel": AverageMeter(),
                        f"mae_{sat_c}_pixel": AverageMeter(),
                        f"bias_{sat_c}_pixel": AverageMeter(),
                        f"iou_{sat_c}_pixel": AverageMeter(),
                        f"ssim_{sat_c}_pixel": AverageMeter(),

                        f"rmse_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"mae_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"bias_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"iou_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"ssim_{sat_c}_to_gpm_pixel": AverageMeter(),

                    }
                    avg_meter_dict_wrt_gt[i] = temp_dict


                binary_accumulators_gt = {
                    "gpm":        CountAccumulator(),
                    f"{sat_c}":        CountAccumulator(),
                    f"{sat_c}_to_gpm": CountAccumulator(),
                }

                avg_meter_dict_wrt_gpm = {}
                for i in range(len(rr_interval_arr)):
                    temp_dict = {
                        "rmse_gpm_pixel": AverageMeter(),
                        "mae_gpm_pixel": AverageMeter(),
                        "bias_gpm_pixel": AverageMeter(),
                        "iou_gpm_pixel": AverageMeter(),
                        "ssim_gpm_pixel": AverageMeter(),
                        
                        f"rmse_{sat_c}_pixel": AverageMeter(),
                        f"mae_{sat_c}_pixel": AverageMeter(),
                        f"bias_{sat_c}_pixel": AverageMeter(),
                        f"iou_{sat_c}_pixel": AverageMeter(),
                        f"ssim_{sat_c}_pixel": AverageMeter(),

                        f"rmse_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"mae_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"bias_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"iou_{sat_c}_to_gpm_pixel": AverageMeter(),
                        f"ssim_{sat_c}_to_gpm_pixel": AverageMeter(),

                     
                    }
                    avg_meter_dict_wrt_gpm[i] = temp_dict


                binary_accumulators_gpm = {
                    "gpm":        CountAccumulator(),
                    f"{sat_c}":        CountAccumulator(),
                    f"{sat_c}_to_gpm": CountAccumulator(),
                }


               
                dict_arr = pickle_load(dict_path)
                k_kept_ds = self.cfg["dataset"]["k_kept"]
                print("this is the k_kept_ds", k_kept_ds)
                if k_kept_ds > 0:
                    dict_arr = dict_arr[:k_kept_ds]

                

                for i, elt in enumerate(dict_arr):
                    if i > 100:
                        do_plot = 0


                    if debug:
                        if i >= 2:
                            break


                    op_idx = i
                    c_target = None
                    if "c_target_override" in cfg: c_target = cfg["c_target_override"]

                    if (op_idx != c_target) and (c_target != None):
                        continue

                    exp_name=f"{res_dir}/{sat_c}/{cfg['model_used']}_{solver}_{nfe}_{t_end}"
                    statistics_file_pixel = os.path.join(exp_name, "", f"000___statistics_pixel.txt")


                    sat_folder_full = f"{exp_name}/images/{op_idx}"
                    os.makedirs(sat_folder_full, exist_ok=True)

                    lon_gpm, lat_gpm, data_gpm = elt["gpm"]["lon"], elt["gpm"]["lat"], elt["gpm"]["data"]
                    lon_f18, lat_f18, data_f18 = elt[sat_c]["lon"], elt[sat_c]["lat"], elt[sat_c]["data"]



                    data_rr = data_gpm[4:6].to(device)
                    rr_gt = data_rr[0:1].unsqueeze(0).to(device)
                    data_gpm = data_gpm[:4]

                    data_gpm = data_gpm.unsqueeze(0).to(device)
                    data_f18 = data_f18.unsqueeze(0).to(device)


     

                    y_gpm = (torch.ones(1) * LABEL_GPM).to(torch.int64)
                    y_f18 = (torch.ones(1) * C_LABEL).to(torch.int64)

                    x_gpm, y_gpm = dl_test.dataset.get_normalized_data_and_labels(device, x=data_gpm, y=y_gpm)
                    x_f18, y_f18 = dl_test.dataset.get_normalized_data_and_labels(device, x=data_f18, y=y_f18)

                    current_sat = "gpm"
                    x_show = dl_train.dataset.get_unormalized_data_and_labels(device, x_gpm, y_gpm)[0]
                    x_show = flatten_channels_and_stack(x_show, stack_vertically=True)
                    file = os.path.join(sat_folder_full, f"{current_sat}_og_images.png")
                    if do_plot:
                        save_2D_tensor_image_as_png(file, x_show)

                    current_sat = sat_c
                    x_show = dl_train.dataset.get_unormalized_data_and_labels(device, x_f18, y_f18)[0]
                    x_show = flatten_channels_and_stack(x_show, stack_vertically=True)
                    file = os.path.join(sat_folder_full, f"{current_sat}_og_images.png")
                    if do_plot:
                        save_2D_tensor_image_as_png(file, x_show)


                    # Perform the domain transfer
                    # ----------------------------------------------------------------------------------------------------------------


                    x_f18_to_gpm, = self.encode_decode_samples(
                        model=model_for_sampling, FM=self.FM, x=x_f18, y1=y_f18, y2=torch.full_like(y_f18, LABEL_GPM), t_enc=t_end,
                        num_steps=nfe, method=self.solver, atol=1e-4, rtol=1e-4,
                        cfg_scale_forward=1.0, cfg_scale_reverse=1.0,
                        do_stochastic_noising=do_stochastic_noising,
                        cut_into_patches=True, overlap_size=overlap_size, overlap_mode=overlap_mode,
                    )
    
                    x_f18_to_gpm = dl_train.dataset.get_unormalized_data_and_labels(device, x_f18_to_gpm, y_gpm)[0].unsqueeze(0)

                   
                 

                    x_show = flatten_channels_and_stack(x_f18_to_gpm, stack_vertically=True)
                    file = os.path.join(sat_folder_full, f"{sat_c}_to_gpm_images.png")
                    if do_plot:
                        save_2D_tensor_image_as_png(file, x_show)


                    x_gpm_to_f18 = self.encode_decode_samples(
                        model=model_for_sampling, FM=self.FM, x=x_gpm, y1=y_gpm, y2=torch.full_like(y_gpm, C_LABEL), t_enc=t_end,
                        num_steps=nfe, method=self.solver, atol=1e-4, rtol=1e-4,
                        cfg_scale_forward=1.0, cfg_scale_reverse=1.0,
                        do_stochastic_noising=do_stochastic_noising,
                        cut_into_patches=True, overlap_size=overlap_size, overlap_mode=overlap_mode,
                    )

                    x_gpm_to_f18 = dl_train.dataset.get_unormalized_data_and_labels(device, x_gpm_to_f18, y_f18)[0]
                    x_show = flatten_channels_and_stack(x_gpm_to_f18, stack_vertically=True)
                    file = os.path.join(sat_folder_full, f"gpm_to_{sat_c}_images.png")
                    if do_plot:
                        save_2D_tensor_image_as_png(file, x_show)

                    x_gpm = dl_train.dataset.get_unormalized_data_and_labels(device, x_gpm, y_gpm)[0]
                    x_f18 = dl_train.dataset.get_unormalized_data_and_labels(device, x_f18, y_f18)[0]

                    rr_gpm = ddrain.forward_test(x_gpm, rr_threshold=rr_threshold)
                    rr_f18 = ddrain.forward_test(x_f18, rr_threshold=rr_threshold)
                    rr_f18_to_gpm = ddrain.forward_test(x_f18_to_gpm, rr_threshold=rr_threshold)
                    
                

                    # Collocate all rrs along the gpm 
                    # ---------------------------------------------------------------------------
                    rr_f18_colloc = collocate_swaths(
                        lon_f18, lat_f18, rr_f18,
                        lon_gpm, lat_gpm, 
                        radius_of_influence=C_ROI,
                        neighbors=C_NEIGHBORS,
                        power=power,
                        fill_value=fill_value,
                    )
                    rr_f18_to_gpm_colloc = collocate_swaths(
                        lon_f18, lat_f18, rr_f18_to_gpm,
                        lon_gpm, lat_gpm, 
                        radius_of_influence=C_ROI,
                        neighbors=C_NEIGHBORS,
                        power=power,
                        fill_value=fill_value,
                    )
                    rr_gpm_colloc = rr_gpm.clone()
                    rr_gt_colloc = rr_gt.clone()
                    # ---------------------------------------------------------------------------
                    
                    # in gt
                    rr_gt_colloc_og = rr_gt_colloc.clone()
                    rr_gpm_colloc_og = rr_gpm_colloc.clone()
                    rr_f18_colloc_og = rr_f18_colloc.clone()
                    rr_f18_to_gpm_colloc_og = rr_f18_to_gpm_colloc.clone()
                    
                    
                    if do_plot:
                        os.makedirs(sat_folder_full, exist_ok=True)
                        titles_arr = ["Ground Truth", "DRAIN on GPM",  f"DRAIN on adapted {sat_c.upper()}", f"DRAIN on {sat_c.upper()}"]
                        lon_arr = [lon_gpm, lon_gpm, lon_f18, lon_f18]
                        lat_arr = [lat_gpm, lat_gpm, lat_f18, lat_f18]
                        rr_arr = [rr_gt, rr_gpm, rr_f18_to_gpm, rr_f18]

                        extent = [lon_f18.min(), lon_f18.max(), lat_f18.min(), lat_f18.max()]

                        save_path = os.path.join(sat_folder_full, "", "rr_comparisons.png")
                        plot_DPR_rainrate_multi(
                            titles=titles_arr,
                            lons=lon_arr,
                            lats=lat_arr,
                            rains=rr_arr,
                            extent = extent,
                            save_path=save_path,
                        )

                    if do_plot:
                        os.makedirs(sat_folder_full, exist_ok=True)
                        titles_arr = ["Ground Truth", "DRAIN on GPM",  f"DRAIN on adapted {sat_c.upper()}", f"DRAIN on {sat_c.upper()}"]
                        lon_arr = [lon_gpm, lon_gpm, lon_gpm, lon_gpm]
                        lat_arr = [lat_gpm, lat_gpm, lat_gpm, lat_gpm]
                        rr_arr = [rr_gt_colloc, rr_gpm_colloc, rr_f18_to_gpm_colloc, rr_f18_colloc]

                        extent = [lon_gpm.min(), lon_gpm.max(), lat_gpm.min(), lat_gpm.max()]
                        
                        save_path = os.path.join(sat_folder_full, "", "rr_comparisons_colloc.png")
                        plot_DPR_rainrate_multi(
                            titles=titles_arr,
                            lons=lon_arr,
                            lats=lat_arr,
                            rains=rr_arr,
                            extent = extent,
                            save_path=save_path,
                            lon_trace_override=lon_f18,
                            lat_trace_override=lat_f18,
                            save_individual=False,
                        )



                    gt_arr = [
                        "gt",
                        "gpm"
                    ]
                    for gt_idx, gt in enumerate(gt_arr):
                        # We Compare with ground truth corra/dpr
                        if gt_idx == 0:
                            rr_gt_colloc = rr_gt_colloc.flatten().cpu()
                            rr_gpm_colloc = rr_gpm_colloc.flatten().cpu()
                            rr_f18_colloc = rr_f18_colloc.flatten().cpu()
                            rr_f18_to_gpm_colloc = rr_f18_to_gpm_colloc.flatten().cpu()

                            
                            valid_mask = ~torch.isnan(rr_f18_colloc) & (rr_gt_colloc >= 0.0)
                            n_pixel_colloc_arr_gt.append(int(valid_mask.sum().item()))

                        # We consider the DRAIN model prediction to be the ground truth
                        elif gt_idx == 1:
                            # The big change is here 
                            # ------------------------------------------------------------------------
                            rr_gt_colloc = rr_gpm_colloc.flatten().cpu()
                            # ------------------------------------------------------------------------
                            rr_gpm_colloc = rr_gpm_colloc.flatten().cpu()
                            rr_f18_colloc = rr_f18_colloc.flatten().cpu()
                            rr_f18_to_gpm_colloc = rr_f18_to_gpm_colloc.flatten().cpu()

                            
                            valid_mask = ~torch.isnan(rr_f18_colloc)
                            n_pixel_colloc_arr_gpm.append(int(valid_mask.sum().item()))

                   

                            
                        if do_plot: 
                            if gt == "gt":
                                rr_gt_colloc_clone = rr_gt_colloc.clone()
                                rr_gpm_colloc_clone = rr_gpm_colloc.clone()
                                rr_f18_colloc_clone = rr_f18_colloc.clone()
                                rr_f18_to_gpm_colloc_clone = rr_f18_to_gpm_colloc.clone()

                                rr_gt_colloc_clone[~valid_mask] = float('nan')
                                rr_gpm_colloc_clone[~valid_mask] = float('nan')
                                rr_f18_colloc_clone[~valid_mask] = float('nan')
                                rr_f18_to_gpm_colloc_clone[~valid_mask] = float('nan')

                                og_shape = rr_gt_colloc_og.shape

                                rr_gt_colloc_clone         = rr_gt_colloc_clone.reshape(og_shape)
                                rr_gpm_colloc_clone        = rr_gpm_colloc_clone.reshape(og_shape)
                                rr_f18_colloc_clone        = rr_f18_colloc_clone.reshape(og_shape)
                                rr_f18_to_gpm_colloc_clone = rr_f18_to_gpm_colloc_clone.reshape(og_shape)
                                                
                                # --- do the same for lon / lat ---
                                lon_flat = lon_gpm.flatten().clone().float()
                                lat_flat = lat_gpm.flatten().clone().float()

                                # extent from VALID coords only (before NaN-ing, so min/max are clean)
                                lon_valid = lon_flat[valid_mask]
                                lat_valid = lat_flat[valid_mask]
                                extent = [lon_valid.min().item(), lon_valid.max().item(),
                                        lat_valid.min().item(), lat_valid.max().item()]

                                # NaN-masked coords, reshaped back to the grid
                                lon_clone = lon_flat.clone()
                                lat_clone = lat_flat.clone()
                                lon_clone[~valid_mask] = float('nan')
                                lat_clone[~valid_mask] = float('nan')
                                lon_clone = lon_clone.reshape(og_shape)
                                lat_clone = lat_clone.reshape(og_shape)

                                os.makedirs(sat_folder_full, exist_ok=True)
                                titles_arr = ["Ground Truth", "DRAIN on GPM", f"DRAIN on adapted {sat_c.upper()}", f"DRAIN on {sat_c.upper()}"]
                                lon_arr = [lon_clone, lon_clone, lon_clone, lon_clone]
                                lat_arr = [lat_clone, lat_clone, lat_clone, lat_clone]
                                rr_arr = [rr_gt_colloc_clone, rr_gpm_colloc_clone, rr_f18_to_gpm_colloc_clone, rr_f18_colloc_clone]


                                save_path = os.path.join(sat_folder_full, "", "rr_comparisons_colloc_gt.png")
                                plot_DPR_rainrate_multi(
                                    titles=titles_arr,
                                    lons=lon_arr,
                                    lats=lat_arr,
                                    rains=rr_arr,
                                    extent = extent,
                                    save_path=save_path,
                                    lon_trace_override=None,
                                    lat_trace_override=None,
                                    save_individual=False,
                                    c1=(0.9, 0.9, 0.9),
                                    c2=(0.9, 0.9, 0.9),
                                    alpha_val=0.1,                                                                                                                                                                                                                                                                      
                                )


                        # Index all the rain from the ground truth
                        rr_gt_colloc = rr_gt_colloc[valid_mask]
                        rr_gpm_colloc = rr_gpm_colloc[valid_mask]
                        rr_f18_colloc = rr_f18_colloc[valid_mask]
                        rr_f18_to_gpm_colloc = rr_f18_to_gpm_colloc[valid_mask]


                        rr_gt_colloc[rr_gt_colloc < rr_threshold] = 0.0
                        rr_gpm_colloc[rr_gpm_colloc < rr_threshold] = 0.0
                        rr_f18_colloc[rr_f18_colloc < rr_threshold] = 0.0
                        rr_f18_to_gpm_colloc[rr_f18_to_gpm_colloc < rr_threshold] = 0.0

                        kept_idx_arr = []
                        for i, elt in enumerate(rr_interval_arr):
                            if isinstance(elt, tuple):
                                r_start, r_end = elt
                                kept_idx = ((rr_gt_colloc >= r_start) & (rr_gt_colloc < r_end)).nonzero().flatten()
                                kept_idx_arr.append(kept_idx)

                        

                        def calculate_all_metrics_gt(rr_gt_colloc, rr_gpm_colloc, rr_f18_colloc, rr_f18_to_gpm_colloc, kept_idx, avg_meters, sat_c="f18"):
                            n_pixel = len(kept_idx)
                            if n_pixel > 0:
                                rr_gt_colloc__ = rr_gt_colloc[kept_idx]
                                rr_gpm_colloc__ = rr_gpm_colloc[kept_idx]
                                rr_f18_colloc__ = rr_f18_colloc[kept_idx]
                                rr_f18_to_gpm_colloc__ = rr_f18_to_gpm_colloc[kept_idx]
                                
                                diff_gpm = rr_gt_colloc__ - rr_gpm_colloc__
                                diff_f18 = rr_gt_colloc__ - rr_f18_colloc__
                                diff_f18_to_gpm = rr_gt_colloc__ - rr_f18_to_gpm_colloc__

                                bias_gpm = diff_gpm.mean()
                                bias_f18 = diff_f18.mean()
                                bias_f18_to_gpm = diff_f18_to_gpm.mean()

                                avg_meters["bias_gpm_pixel"].update(bias_gpm.item(), n_pixel)
                                avg_meters[f"bias_{sat_c}_pixel"].update(bias_f18.item(), n_pixel)
                                avg_meters[f"bias_{sat_c}_to_gpm_pixel"].update(bias_f18_to_gpm.item(), n_pixel)
                                
                                mae_gpm = diff_gpm.abs().mean()
                                mae_f18 = diff_f18.abs().mean()
                                mae_f18_to_gpm = diff_f18_to_gpm.abs().mean()
                                
                                avg_meters["mae_gpm_pixel"].update(mae_gpm.item(), n_pixel)
                                avg_meters[f"mae_{sat_c}_pixel"].update(mae_f18.item(), n_pixel)
                                avg_meters[f"mae_{sat_c}_to_gpm_pixel"].update(mae_f18_to_gpm.item(), n_pixel)

                                rmse_gpm = torch.pow((diff_gpm**2).mean(), 0.5)
                                rmse_f18 = torch.pow((diff_f18**2).mean(), 0.5)
                                rmse_f18_to_gpm = torch.pow((diff_f18_to_gpm**2).mean(), 0.5)

                                avg_meters["rmse_gpm_pixel"].update(rmse_gpm.item(), n_pixel)
                                avg_meters[f"rmse_{sat_c}_pixel"].update(rmse_f18.item(), n_pixel)
                                avg_meters[f"rmse_{sat_c}_to_gpm_pixel"].update(rmse_f18_to_gpm.item(), n_pixel)

                        def calculate_binary_metrics(rr_gt_colloc, rr_gpm_colloc, rr_f18_colloc, rr_f18_to_gpm_colloc, accumulators, sat_c="f18"):
                            if len(rr_gt_colloc) == 0:
                                return

                            def binarize(t):
                                return (t > 0.0).detach().cpu().numpy().astype(np.int8).ravel()

                            gt = binarize(rr_gt_colloc)
                            accumulators["gpm"].update(gt,        binarize(rr_gpm_colloc))
                            accumulators[f"{sat_c}"].update(gt,        binarize(rr_f18_colloc))
                            accumulators[f"{sat_c}_to_gpm"].update(gt, binarize(rr_f18_to_gpm_colloc))


                        if gt == "gt":
                            current_avg_meter_dict = avg_meter_dict_wrt_gt
                            current_binary_accumulator = binary_accumulators_gt
                        elif gt == "gpm":
                            current_avg_meter_dict = avg_meter_dict_wrt_gpm
                            current_binary_accumulator = binary_accumulators_gpm
                        else:
                            raise ValueError("Should not reach this condition here")
                        
                        

                        for j, kept_idx in enumerate(kept_idx_arr):
                            calculate_all_metrics_gt(
                                rr_gt_colloc, rr_gpm_colloc, rr_f18_colloc, rr_f18_to_gpm_colloc,
                                kept_idx=kept_idx, avg_meters=current_avg_meter_dict[j],
                                sat_c=sat_c,
                            )   
                        calculate_binary_metrics(
                            rr_gt_colloc, rr_gpm_colloc, rr_f18_colloc, rr_f18_to_gpm_colloc,
                            current_binary_accumulator,
                            sat_c=sat_c,
                        )                        
                        # ------------------------------------------------------------------------------------------------
                        # End of calculations with respect to the ground truth CORRA / DPR



                mode_arr = [
                    "corra_gt",
                    "gpm_gt",
                ]

                for mode in mode_arr:
                    if mode == "corra_gt":
                        avg_meter_dict_c = avg_meter_dict_wrt_gt
                        binary_accumulators_c = binary_accumulators_gt
                        n_pixels = sum(n_pixel_colloc_arr_gt)
                    elif mode == "gpm_gt":
                        avg_meter_dict_c = avg_meter_dict_wrt_gpm
                        binary_accumulators_c = binary_accumulators_gpm
                        n_pixels = sum(n_pixel_colloc_arr_gpm)
                    else:
                        raise ValueError(f"Do not recognize this mode {mode} here, check the array above")


                    # Build column headers from rr_interval_arr
                    col_headers = []
                    for elt in rr_interval_arr:
                        if isinstance(elt, tuple):
                            r_start, r_end = elt
                            col_headers.append(f"[{r_start},{r_end}]")
                        else:
                            col_headers.append("union_all")

                    entries = ["gpm", f"{sat_c}", f"{sat_c}_to_gpm"]
                    metrics = ["rmse", "mae", "bias"]

                    row_label_width = max(len(f"{m}_{e}") for m in metrics for e in entries)
                    row_label_width = max(row_label_width, len("metric/rr")) + 2
                    decimals = 4

                    base, _ = os.path.splitext(statistics_file_pixel)

                    for metric in metrics:
                        # Compute per-column width: max of header and all formatted values in that column
                        col_widths = []
                        for i, h in enumerate(col_headers):
                            max_val_len = 0
                            for entry in entries:
                                key = f"{metric}_{entry}_pixel"
                                val = avg_meter_dict_c[i][key].avg
                                max_val_len = max(max_val_len, len(f"{val:.{decimals}f}"))
                            col_widths.append(max(len(h), max_val_len) + 2)

                        lines = []
                        header = "metric/rr".ljust(row_label_width) + "".join(h.rjust(w) for h, w in zip(col_headers, col_widths))
                        lines.append(header)
                        lines.append("-" * len(header))

                        for entry in entries:
                            key = f"{metric}_{entry}_pixel"
                            row = f"{metric}_{entry}".ljust(row_label_width)
                            for i, w in enumerate(col_widths):
                                val = avg_meter_dict_c[i][key].avg
                                row += f"{val:>{w}.{decimals}f}"
                            lines.append(row)

                        lines.append("")
                        lines.append(f"n_pixel_colloc_{mode}: {n_pixels}")

                        out_path = f"{base}_{mode}_{metric}.txt"
                        with open(out_path, "w") as f:
                            f.write("\n".join(lines) + "\n")
                        print(f"Wrote {out_path}")
                        print("\n".join(lines))
                        print()



                    metrics = ["iou", "recall", "far", "bias", "precision"]
                    decimals = 4


                    def get_metric(acc, name):
                        return {"iou": acc.iou(), "recall": acc.recall(),
                                "far": acc.far(), "precision": acc.precision(),
                                "bias": acc.bias()}[name]

                    # Sanity check — exit if any metric is NaN
                    for entry in entries:
                        for metric in metrics:
                            val = get_metric(binary_accumulators_c[entry], metric)
                            if math.isnan(val) or math.isinf(val):
                                acc = binary_accumulators_c[entry]
                                sys.exit(f"[binary metrics] {metric}_{entry}={val}. "
                                        f"tp={acc.tp} fp={acc.fp} fn={acc.fn}")

                    row_label_width = max(max(len(e) for e in entries), len("entry/metric")) + 2
                    col_widths = [
                        max(len(m), max(len(f"{get_metric(binary_accumulators_c[e], m):.{decimals}f}") for e in entries)) + 2
                        for m in metrics
                    ]

                    lines = ["entry/metric".ljust(row_label_width) + "".join(m.rjust(w) for m, w in zip(metrics, col_widths))]
                    lines.append("-" * len(lines[0]))
                    for entry in entries:
                        row = entry.ljust(row_label_width)
                        for metric, w in zip(metrics, col_widths):
                            row += f"{get_metric(binary_accumulators_c[entry], metric):>{w}.{decimals}f}"
                        lines.append(row)

                    base, _ = os.path.splitext(statistics_file_pixel)
                    out_path = f"{base}_{mode}__binary.txt"
                    with open(out_path, "w") as f:
                        f.write("\n".join(lines) + "\n")
                    print(f"Wrote {out_path}")
                    print("\n".join(lines))


