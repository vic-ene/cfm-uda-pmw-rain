import torch
from torch import nn
import math

def freeze(model):
    """Put model in eval mode and freeze all parameters."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def ema_fn(source, target, decay=0.9999):
    for p_tgt, p_src in zip(target.parameters(), source.parameters()):
        # Equivalent to: p_tgt = p_tgt * decay + p_src * (1 - decay)
        p_tgt.mul_(decay).add_(p_src, alpha=1 - decay)



def count_params_millions(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    to_m = lambda x: x / 1e6  # convert to millions

    print(f"Total parameters:       {to_m(total):.3f} M")
    print(f"Trainable parameters:   {to_m(trainable):.3f} M")
    print(f"Non-trainable params:   {to_m(non_trainable):.3f} M")

    return total




