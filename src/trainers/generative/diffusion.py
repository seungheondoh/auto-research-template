"""DDPM denoising diffusion probabilistic model with DiT backbone.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020).
Backbone:  Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICLR 2023).

Supports epsilon, x_start, and v_prediction parameterisations.

Batch keys:
    image [B, C, H, W]
    label [B]            (optional, for class-conditional generation)
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.dit import DiT
from src.modules.ema import EMA
from src.modules.noise_scheduler import DDPMScheduler


class DiffusionTrainer(BaseTrainer):
    """DDPM trainer.

    Config keys:
        img_size, patch_size, in_channels, hidden_size, depth, num_heads — DiT
        num_classes:      0 for unconditional, >0 for class-conditional.
        num_timesteps:    diffusion steps T (default 1000).
        noise_schedule:   'cosine' | 'linear' (default 'cosine').
        prediction_type:  'epsilon' | 'x_start' | 'v_prediction' (default 'epsilon').
        use_ema:          maintain EMA copy for inference (default True).
        ema_decay:        EMA decay rate (default 0.9999).
        attn_impl:        'sdpa' | 'fmha' | 'flex_attention' (default 'sdpa').
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.backbone = DiT(cfg)
        self.scheduler = DDPMScheduler(
            num_timesteps=cfg.get("num_timesteps", 1000),
            schedule=cfg.get("noise_schedule", "cosine"),
            beta_start=cfg.get("beta_start", 1e-4),
            beta_end=cfg.get("beta_end", 0.02),
        )
        self.pred_type = cfg.get("prediction_type", "epsilon")
        self.ema = EMA(self.backbone, cfg.get("ema_decay", 0.9999)) if cfg.get("use_ema", True) else None

    def _loss(self, batch) -> torch.Tensor:
        x0 = batch["image"]
        y  = batch.get("label")
        T  = self.cfg.get("num_timesteps", 1000)
        t  = torch.randint(0, T, (x0.shape[0],), device=self.device)
        noise = torch.randn_like(x0)
        xt    = self.scheduler.q_sample(x0, t, noise, self.device)
        pred  = self.backbone(xt, t, y)
        target = self.scheduler.get_target(x0, noise, t, self.pred_type, self.device)
        return F.mse_loss(pred, target)

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
