import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.lazy_utils import *
from utils.my_nn_utils import freeze


import torchvision
import matplotlib.pyplot as plt
import math

import warnings
import torchdiffeq

from torchcfm.optimal_transport import OTPlanSampler


ODE_INVERSION_ARR = [
    "cfm",
    "ot_cfm",
]


ODE_FULL_ARR = [
    "cfm_d2d", 
    "ot_cfm_d2d",
    "si",
    "si_cos_sq",
]


def logit_randn(*size, mu=0.0, std=1.0, **kwargs):
    z = torch.randn(*size, **kwargs)
    return torch.sigmoid(mu + std * z)


def interp_image_and_noise(x, t_value=0.5):
    noise = torch.randn_like(x)
    t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
    t = t.view(-1, *([1] * (x.ndim - 1)))
    x_t = t * x + (1 - t) * noise
    return x_t


class MergedCFM:
    def __init__(
        self,
        time_sampler="uniform",
        noise_type="gaussian",
        cfm_used="cfm",
        logit_normal_mu_t=0.0,
        logit_normal_std_t=1.0,
        si_gamma_fn="sin_sq",
        si_gamma_a=2,
        target_label=0,
        model_pretrain=None,
        cfg=None,
        device=None,
    ):
        if device is None:
            raise ValueError("Need to pass in the device for the MergedCFM")

        if time_sampler not in ("uniform", "logit_normal"):
            raise ValueError(f"Unknown time_sampler: {time_sampler}")

        initialize_vars(self, locals(), exclude=[])
        self.cfg = cfg

        self.device = device
        self.cond_used = self.cfg["cond_used"]


        if self.si_gamma_fn == "sin_sq" and self.si_gamma_a != 1:
            warnings.warn(
                "sin_sq gamma requires coefficient 1 for a standard-Gaussian "
                "midpoint; overriding si_gamma_a from "
                f"{self.si_gamma_a} to 1."
            )
            self.si_gamma_a = 1

        self.ot_sampler = OTPlanSampler()


    # ------------------------------------------------------------------
    # Training-time sampling of (t, xt, target)
    # ------------------------------------------------------------------
    def sample_t(self, batch_size, ref_type, ndim=None):
        if self.time_sampler == "uniform":
            t = torch.rand(batch_size).type_as(ref_type)
        else:
            t = logit_randn(
                batch_size, mu=self.logit_normal_mu_t, std=self.logit_normal_std_t
            )
        if ndim is not None:
            t = t.reshape(-1, *([1] * (ndim - 1)))
        return t

    def sample_noise(self, x):
        device = x.device
        if self.noise_type == "gaussian":
            z = torch.randn_like(x)
            return z.to(device)
        else:
            raise ValueError("did not implement this noise type")

    def gamma_fn_and_derivative(self, t):
        a = self.si_gamma_a
        if self.si_gamma_fn == "sqrt_t_times_1_sub_t":
            gamma = torch.sqrt(a * t * (1 - t))
            d_gamma = (a * (1 - 2 * t)) / (2 * gamma)
            return gamma, d_gamma
        elif self.si_gamma_fn == "t_times_1_sub_t":
            gamma = a * t * (1 - t)
            d_gamma = a * (1 - 2 * t)
            return gamma, d_gamma
        elif self.si_gamma_fn == "sin_sq":
            gamma = a * torch.sin(math.pi * t) ** 2
            d_gamma = a * math.pi * torch.sin(2 * math.pi * t)
            return gamma, d_gamma
        else:
            raise ValueError(f"Do not recognize the gamma_fn {self.si_gamma_fn}")

    def sample_location_and_conditional_flow(self, x, y, epoch):

        if self.cfm_used == "cfm":
            t = self.sample_t(x.shape[0], ref_type=x, ndim=x.dim())
            z = self.sample_noise(x)
            xt = t * x + (1 - t) * z
            ut = x - z
            return t, xt, ut, y

        elif self.cfm_used == "ot_cfm":
            t = self.sample_t(x.shape[0], ref_type=x, ndim=x.dim())
            z = self.sample_noise(x)
            z, x, _, y = self.ot_sampler.sample_plan_with_labels(z, x, None, y, replace=False)
            xt = t * x + (1 - t) * z
            ut = x - z
            return t, xt, ut, y



        elif self.cfm_used in ODE_FULL_ARR:
            if y is None:
                raise ValueError("Need to know labels to setup distribution to distribution flow matching")


            target_idx = (y == self.target_label).nonzero(as_tuple=False).flatten()
            other_idx = (y != self.target_label).nonzero(as_tuple=False).flatten()

            x1 = x[target_idx]
            x0 = x[other_idx]
            dupl_factor = len(x0) // len(x1)
            x1 = x1.repeat(dupl_factor, 1, 1, 1)

            # We learn to map all distributions with label different from label 0 to label 0
            t = self.sample_t(x1.shape[0], ref_type=x1, ndim=x1.dim())
            y = y[other_idx]

            if self.cfm_used == "cfm_d2d":
                xt = t * x1 + (1 - t) * x0
                ut = x1 - x0
                return t, xt, ut, y

            elif self.cfm_used == "ot_cfm_d2d":
                x0, x1, y, _ = self.ot_sampler.sample_plan_with_labels(
                    x0, x1, y, None, replace=False
                )
                xt = t * x1 + (1 - t) * x0
                ut = x1 - x0
                return t, xt, ut, y

            elif self.cfm_used == "si":
                z = self.sample_noise(x1)
                gamma, d_gamma = self.gamma_fn_and_derivative(t)
                xt = t * x1 + (1 - t) * x0 + gamma * z
                ut = x1 - x0 + d_gamma * z
                return t, xt, ut, y

            elif self.cfm_used == "si_cos_sq":
                z = self.sample_noise(x1)

                cos_sq = torch.cos(math.pi * t) ** 2
                d_cos_sq = -math.pi * torch.sin(2 * math.pi * t)

                enc = (t < 0.5).type_as(t)
                dec = (t > 0.5).type_as(t)

                alpha, beta = cos_sq * enc, cos_sq * dec
                d_alpha, d_beta = d_cos_sq * enc, d_cos_sq * dec

                gamma, d_gamma = self.gamma_fn_and_derivative(t)

                xt = alpha * x0 + beta * x1 + gamma * z
                ut = d_alpha * x0 + d_beta * x1 + d_gamma * z
                return t, xt, ut, y

        else:
            raise ValueError(f"Did not implement for {self.cfm_used}")

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def get_loss(self, model, x, y, epoch):
        t, xt, ut, y = self.sample_location_and_conditional_flow(x, y, epoch)

        pred = model(t, xt, y=y)
        target = ut

        loss = torch.mean((pred - target) ** 2)
        return {"loss": loss}



    def ode_field(self, model_, t, x, y, cfg_scale=1.0):
        if self.cond_used == "vanilla":
            return model_.forward(t, x, y)


        elif self.cond_used == "cfg":
            y_cond = y
            y_uncond = torch.ones_like(y) * (model_.y_embedder.cfg_label) # Retrieve label associated to cfg 

            vel_cond = model_.forward(t, x, y_cond)
            vel_uncond = model_.forward(t, x, y_uncond)
            vel = cfg_scale * vel_cond + (1 - cfg_scale) * vel_uncond
            return vel
            
        else:
            raise ValueError("Do not recognise the conditioning for sampling")


  

    @torch.inference_mode()
    def flow_integrate_torchdiffeq(self, model, x, y, t_start=0.0, t_end=1.0, num_steps=1, method="dopri5", atol=1e-4, rtol=1e-4, cfg_scale=1.0):
        model_ = model.module if hasattr(model, "module") else model
        was_training = model_.training
        model_.eval()

        try:
            traj = torchdiffeq.odeint(
                lambda t, x_t: self.ode_field(model_, t, x_t, y, cfg_scale),
                x,
                torch.linspace(t_start, t_end, 2, device=x.device),
                atol=atol,
                rtol=rtol,
                method=method,
                options=(dict(step_size=((abs(t_start - t_end)) / num_steps)) if method in {"euler", "rk4", "midpoint", "explicit_adams", "implicit_adams"} else None),
            )[-1]
        finally:
            model_.train(was_training)

        return traj



    @torch.inference_mode()
    def adapt_torchdiffeq(self, model, x, y1, y2, t_enc=0.5, num_steps=1, method="dopri5", atol=1e-4, rtol=1e-4,
                            cfg_scale_forward=1.0, cfg_scale_reverse=1.0,
                            do_stochastic_noising=0, cfm_used_override=None
                            ):
        
        cfm_used = self.cfg["cfm_used"] if cfm_used_override is None else cfm_used_override


        # Adapt by integrating full ode from 0 to 1 / from source to target
        if cfm_used in ODE_FULL_ARR:
            x = self.flow_integrate_torchdiffeq(
                model, x, y1, t_start=0.0, t_end=1.0, num_steps=num_steps, method=method, atol=atol, rtol=rtol, cfg_scale=cfg_scale_forward
            )
            return x
            
        
        # Adapt by doing inversion and changing label
        elif cfm_used in ODE_INVERSION_ARR:
            x_og = x.clone()


            # Encode image using ODE
            x = self.flow_integrate_torchdiffeq(
                model, x, y1, t_start=1.0, t_end=t_enc, num_steps=num_steps, method=method, atol=atol, rtol=rtol, cfg_scale=cfg_scale_forward,
            )

            # Add option to noise image using some random gaussian noise plus an interpolation (stochastic)
            if do_stochastic_noising: 
                x = interp_image_and_noise(x_og, t_value=t_enc)

            # Decode image using ODE
            x = self.flow_integrate_torchdiffeq(
                model, x, y2, t_start=t_enc, t_end=1.0, num_steps=num_steps, method=method, atol=atol, rtol=rtol, cfg_scale=cfg_scale_reverse,
            )
            return x

       
        else:
            raise ValueError("Did not plan for this")                
           



