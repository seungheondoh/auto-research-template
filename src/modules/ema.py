from copy import deepcopy
import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model parameters.

    Used by: JEPA target encoder, diffusion / flow / drift inference,
             VQ-VAE commitment stabilization.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.shadow = deepcopy(model).eval()
        self.decay = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.lerp_(p.data, 1.0 - self.decay)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
