"""InfoNCE / NT-Xent contrastive representation learning with ViTEncoder backbone.

Batch keys:
    image [B, C, H, W]   — single image per sample (augmented in-trainer)
    view1, view2 [B, C, H, W]  — pre-computed augmented views (preferred)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.dit import ViTEncoder
from src.modules.losses import info_nce_loss


class _Projector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class InfoNCETrainer(BaseTrainer):
    """InfoNCE contrastive trainer.

    Config keys:
        img_size, patch_size, in_channels, hidden_size, depth, num_heads — ViTEncoder
        proj_dim:    projection head output dimension (default 128).
        temperature: InfoNCE temperature τ (default 0.07).
        attn_impl:   'sdpa' | 'fmha' | 'flex_attention' (default 'sdpa').
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.encoder     = ViTEncoder(cfg)
        self.projector   = _Projector(cfg.get("hidden_size", 384), cfg.get("proj_dim", 128))
        self.temperature = cfg.get("temperature", 0.07)

    def _aug(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x.flip(-1) if torch.rand(1).item() > 0.5 else x
            x = (x + torch.randn_like(x) * 0.05).clamp(-1, 1)
        return x

    def _step(self, batch):
        x = batch["image"]
        if "view1" in batch and "view2" in batch:
            v1, v2 = batch["view1"], batch["view2"]
        else:
            v1, v2 = self._aug(x.clone()), self._aug(x.clone())
        z1 = self.projector(self.encoder(v1))
        z2 = self.projector(self.encoder(v2))
        return info_nce_loss(z1, z2, self.temperature)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self._log_lr()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
