"""OT-CFM (Optimal-Transport Conditional Flow Matching) with DiT backbone.

Reference: Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023).
           Tong et al., "Improving and Generalizing Flow-Matching" (NeurIPS 2023).

OT-CFM defines a straight-line path from noise x1 ~ N(0,I) to data x0:
    x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
    v_target = x1 - (1 - sigma_min) * x0

Batch keys:
    image [B, C, H, W]
    label [B]            (optional)
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.dit import DiT
from src.modules.ema import EMA


class FlowMatchingTrainer(BaseTrainer):
    """OT-CFM trainer.

    Config keys:
        img_size, patch_size, in_channels, hidden_size, depth, num_heads — DiT
        num_classes:  0 = unconditional (default 0).
        sigma_min:    noise floor (default 1e-4).
        use_ema:      maintain EMA for inference (default True).
        ema_decay:    default 0.9999.
        attn_impl:    'sdpa' | 'fmha' | 'flex_attention' (default 'sdpa').
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.backbone  = DiT(cfg)
        self.sigma_min = cfg.get("sigma_min", 1e-4)
        self.ema = EMA(self.backbone, cfg.get("ema_decay", 0.9999)) if cfg.get("use_ema", True) else None

    def _loss(self, batch) -> torch.Tensor:
        x0 = batch["image"]
        y  = batch.get("label")
        t  = torch.rand(x0.shape[0], device=self.device)
        x1 = torch.randn_like(x0)
        t_b = t.view(-1, 1, 1, 1)
        xt  = (1 - (1 - self.sigma_min) * t_b) * x0 + t_b * x1
        v_target = x1 - (1 - self.sigma_min) * x0
        t_int = (t * 999).long()
        return F.mse_loss(self.backbone(xt, t_int, y), v_target)

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
