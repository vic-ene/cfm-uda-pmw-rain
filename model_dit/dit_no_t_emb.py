# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math

import sys

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))        # ../
sys.path.append(str(BASE_DIR.parent.parent)) # ../..

from timm.models.vision_transformer import  Attention, Mlp, PatchEmbed
from utils.timm_src_code.patch_embed import PatchEmbed as PatchEmbedMine


from model__it_utils._it_utils import modulate, TimestepEmbedder, LabelEmbedder, get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed

#################################################################################
#                                 Core DiT Model                                #
#################################################################################



def compute_starts(L, s, overlap):
    stride = s - overlap
    assert stride > 0, "overlap must be smaller than patch size"

    starts = list(range(0, max(L - s + 1, 1), stride))

    # force last patch to touch the end
    if starts[-1] != L - s:
        starts.append(L - s)

    return starts
    
def cut_tensor_into_square_overlapping_patches(x, s, overlap_size=1):
    """
    x: Tensor of shape (B, C, H, W) or (C, H, W)
    s: patch size
    overlap_size: overlap in pixels (same in H and W)
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)  # (1, C, H, W)

    B, C, H, W = x.shape
    h_starts = compute_starts(H, s, overlap_size)
    w_starts = compute_starts(W, s, overlap_size)

    patches = []
    for b in range(B):
        patches_b = []
        for i in h_starts:
            for j in w_starts:
                patches_b.append(x[b, :, i:i+s, j:j+s])
        patches.append(torch.stack(patches_b))

    patches = torch.stack(patches)  # (B, N, C, s, s)
    patches = patches.view(-1, C, s, s)  # (B*N, C, s, s)

    meta = {
        "original_shape": (B, C, H, W),
        "h_starts": h_starts,
        "w_starts": w_starts,
        "patch_size": s,
    }

    return patches, meta

def stitch_square_overlapping_patches_back(
    patches,
    meta,
    mode="mean",
):
    """
    patches: (B*N, C, s, s)
    meta: dict returned by cutter
    mode: "mean" or "first"
    """
    B, C, H, W = meta["original_shape"]
    C = patches.shape[1]

    s = meta["patch_size"]
    h_starts = meta["h_starts"]
    w_starts = meta["w_starts"]

    device = patches.device
    dtype = patches.dtype

    output = torch.zeros((B, C, H, W), device=device, dtype=dtype)
    if mode == "mean":
        weight = torch.zeros_like(output)

    N = len(h_starts) * len(w_starts)
    patches = patches.view(B, N, C, s, s)

    for b in range(B):
        idx = 0
        for i in h_starts:
            for j in w_starts:
                patch = patches[b, idx]
                idx += 1

                if mode == "last":
                    output[b, :, i:i+s, j:j+s] = patch
                elif mode == "mean":
                    output[b, :, i:i+s, j:j+s] += patch
                    weight[b, :, i:i+s, j:j+s] += 1
                else:
                    raise ValueError(f"Unknown mode: {mode}")

    if mode == "mean":
        output = output / weight.clamp(min=1)

    return output



class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, is_emb_active=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.is_emb_active = is_emb_active
        if self.is_emb_active:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )


    def forward(self, x, c):
        if self.is_emb_active:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(modulate(self.norm1(x), None, None))
            x = x + self.mlp(modulate(self.norm2(x), None, None))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, is_emb_active=False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        self.is_emb_active = is_emb_active
        if self.is_emb_active:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        if self.is_emb_active:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
            x = self.linear(x)
        else:
            # print("shape of x in final layer", x.shape)     # 64 x 1024 x 384 or 64 x 256 x 384
            x = modulate(self.norm_final(x), None, None)
            x = self.linear(x)
            # print("shape of x after final layer", x.shape)  # 64 x 1024 x 64 or 64 x 256 x 64

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=0,
        learn_sigma=True,
        is_emb_active=False,
        cfg=None,
        t_emb_dropout_label=-1
    ):
        super().__init__()
        self.cfg = cfg
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.t_emb_dropout_label=t_emb_dropout_label
        self.input_size = input_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.patch_embed_used = self.cfg["patch_embed_used"]

        self.is_emb_active = is_emb_active
        if self.is_emb_active:
            self.t_embedder = TimestepEmbedder(hidden_size)
        else:
            self.t_embedder = None
        
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, cond_used="vanilla")

        num_patches = self.x_embedder.num_patches

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, is_emb_active=self.is_emb_active) for _ in range(depth)
        ])

        if self.patch_embed_used == "vanilla":
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, is_emb_active=self.is_emb_active)
        else:
            self.final_layer = nn.Identity()


        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if not self.patch_embed_used in ["learn_extra_embeds"]:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((self.x_embedder.num_patches // 2)** 0.5), cls_token=True, extra_tokens=(self.x_embedder.num_patches // 2))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.patch_embed_used == "vanilla":
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        #nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        if self.y_embedder.num_classes > 0:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=self.cfg["_it"]["label_emb_noise_amplitude"])

        # Initialize timestep embedding MLP:
        if self.t_embedder != None:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if self.is_emb_active:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.patch_embed_used == "vanilla":
            if self.is_emb_active:
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

        def run_custom_init(module):
            if hasattr(module, "my_custom_init"):
                module.my_custom_init()

        self.apply(run_custom_init)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)

        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t=None, y=None):
        x, t = t, x

        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        if self.t_embedder != None:
            t = self.t_embedder(t, b=x.shape[0])        # (N, D)
        else:
            t = 0

        if y != None:
            drop_idx = (y == self.t_emb_dropout_label).nonzero().flatten()
            if len(drop_idx) > 0:
                t[drop_idx] = 0.0
                
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)                
        x = self.unpatchify(x)                  
       
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidforance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


    @torch.no_grad()
    def forward_test(self, x, y=None, rr_threshold=0.1):
        """Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        device = x.device

        self.log_denorm_base =  self.cfg["log"]["base_transform"]

        MIN_TB, MAX_TB = 80, 350
        x = x.clip(MIN_TB, MAX_TB)
        x = (x - MIN_TB) / (MAX_TB - MIN_TB)
        x = (x - 0.5) / 0.5

        h, w = x.shape[-2], x.shape[-1]

        x, meta = cut_tensor_into_square_overlapping_patches(x, self.input_size, overlap_size=1)
        t = torch.ones(x.shape[0]).to(device)
        out = self(t, x, y)
        out = stitch_square_overlapping_patches_back(out, meta)

        out = torch.pow(self.log_denorm_base, 2 * out) - 1.0
        out[out < rr_threshold] = 0.0

        return out


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}



def count_params_millions(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    to_m = lambda x: x / 1e6  # convert to millions

    print(f"Total parameters:       {to_m(total):.3f} M")
    print(f"Trainable parameters:   {to_m(trainable):.3f} M")
    print(f"Non-trainable params:   {to_m(non_trainable):.3f} M")

    return total




if __name__ == "__main__":
    device = torch.device("cuda")
    b, c, h, w = 2, 4, 128, 128
    z = torch.randn(b, c, h, w).to(device)

    model = DiT(depth=8, hidden_size=384, patch_size=8, num_heads=6, input_size=h, in_channels=c, learn_sigma=False).to(device)
    count_params_millions(model)

    t = torch.rand(b).to(device)
    y = torch.zeros(b).to(device).to(torch.int64)

    z = model(t, z, y)
    print(z.shape)

    
   
