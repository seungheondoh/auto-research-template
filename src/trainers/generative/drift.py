"""Stochastic Interpolant / Score SDE drift matching with DiT backbone.

Reference: Albergo & Vanden-Eijnden, "Building Normalizing Flows with
           Stochastic Interpolants" (ICLR 2023).

Interpolant: x_t = alpha(t) * x0 + sigma(t) * noise
Drift target: d/dt x_t = alpha'(t) * x0 + sigma'(t) * noise

Schedules:
    'linear'  — alpha=1-t, sigma=t       (same as OT-CFM)
    'vp'      — alpha=sqrt(1-t^2), sigma=t
    'cosine'  — alpha=cos(π/2·t), sigma=sin(π/2·t)

Batch keys:
    image [B, C, H, W]
    label [B]             (optional)
"""

import math
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.dit import DiT
from src.modules.ema import EMA


def _get_alpha_sigma(t_b: torch.Tensor, schedule: str):
    if schedule == "linear":
        return 1.0 - t_b, -1.0, t_b, 1.0
    if schedule == "vp":
        alpha   = (1 - t_b ** 2).sqrt()
        d_alpha = -t_b / alpha.clamp(min=1e-8)
        return alpha, d_alpha, t_b, 1.0
    if schedule == "cosine":
        alpha   = torch.cos(math.pi / 2 * t_b)
        d_alpha = -math.pi / 2 * torch.sin(math.pi / 2 * t_b)
        sigma   = torch.sin(math.pi / 2 * t_b)
        d_sigma =  math.pi / 2 * torch.cos(math.pi / 2 * t_b)
        return alpha, d_alpha, sigma, d_sigma
    raise ValueError(f"Unknown interpolant '{schedule}'. Choose: linear, vp, cosine.")


class DriftTrainer(BaseTrainer):
    """Stochastic Interpolant drift trainer.

    Config keys:
        img_size, patch_size, in_channels, hidden_size, depth, num_heads — DiT
        interpolant:  'linear' | 'vp' | 'cosine' (default 'linear').
        num_classes:  0 = unconditional (default 0).
        use_ema:      default True.
        ema_decay:    default 0.9999.
        attn_impl:    'sdpa' | 'fmha' | 'flex_attention' (default 'sdpa').
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.backbone = DiT(cfg)
        self.schedule = cfg.get("interpolant", "linear")
        self.ema = EMA(self.backbone, cfg.get("ema_decay", 0.9999)) if cfg.get("use_ema", True) else None

    def _loss(self, batch) -> torch.Tensor:
        x0    = batch["image"]
        y     = batch.get("label")
        noise = torch.randn_like(x0)
        t     = torch.rand(x0.shape[0], device=self.device)
        t_b   = t.view(-1, 1, 1, 1)
        alpha, d_alpha, sigma, d_sigma = _get_alpha_sigma(t_b, self.schedule)
        xt           = alpha * x0 + sigma * noise
        drift_target = d_alpha * x0 + d_sigma * noise
        return F.mse_loss(self.backbone(xt, (t * 999).long(), y), drift_target)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self._log_lr()
        if self.ema is not None:
            self.ema.update(self.backbone)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
