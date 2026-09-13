import os 
import sys 

import torch
import math
import numpy as np




def flatten_channels_and_stack(batch: torch.Tensor, stack_vertically: bool = False) -> torch.Tensor:
    """
    Flatten channels horizontally for each image in the batch, preserving channel info.
    
    Args:
        batch (torch.Tensor): Input tensor of shape (B, C, H, W)
        stack_vertically (bool): If True, stack all images vertically into one tall image.
        
    Returns:
        torch.Tensor: 
            - If stack_vertically=False: shape (B, H, C*W)
            - If stack_vertically=True: shape (B*H, C*W)
    """
    if len(batch.shape) == 3:
        batch = batch.unsqueeze(0)

    # Move channels next to width: (B, C, H, W) -> (B, H, C, W)
    x = batch.permute(0, 2, 1, 3)
    
    # Concatenate channels horizontally: (B, H, C, W) -> (B, H, C*W)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    
    if stack_vertically:
        # Stack all images vertically: (B, H, C*W) -> (B*H, C*W)
        x = x.reshape(-1, x.shape[-1])
    
    return x



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





