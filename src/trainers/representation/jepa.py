"""I-JEPA style Joint Embedding Predictive Architecture.

Reference: Assran et al., "Self-Supervised Learning from Images with a
           Joint-Embedding Predictive Architecture" (CVPR 2023).

Batch keys:
    image [B, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.dit import ViTEncoder, SelfAttnBlock, _make_args
from src.modules.ema import EMA


class _Predictor(nn.Module):
    """Lightweight transformer that predicts target tokens from context tokens."""

    def __init__(self, cfg):
        super().__init__()
        args = _make_args(cfg)
        pred_depth = cfg.get("pred_depth", 4)
        self.blocks = nn.ModuleList([SelfAttnBlock(args) for _ in range(pred_depth)])
        from src.modules.transformers import RMSNorm, RotaryEmbedding
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.rope  = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=cfg.get("max_seqlen", 1024),
        )
        self.attn_impl = cfg.get("attn_impl", "sdpa")

    def forward(self, context: torch.Tensor, num_targets: int) -> torch.Tensor:
        B, _, D = context.shape
        queries  = torch.zeros(B, num_targets, D, device=context.device)
        x        = torch.cat([context, queries], dim=1)
        freq_cis = self.rope(seqlen=x.shape[1])
        for block in self.blocks:
            x = block(x, freq_cis, self.attn_impl)
        return self.norm(x)[:, -num_targets:]


class JEPATrainer(BaseTrainer):
    """I-JEPA trainer.

    Config keys:
        img_size, patch_size, in_channels, hidden_size, depth, num_heads — ViTEncoder
        pred_depth:      predictor depth (default 4).
        context_ratio:   fraction of patches as context (default 0.5).
        target_ratio:    fraction of patches as targets (default 0.25).
        ema_decay:       target encoder momentum (default 0.996).
        attn_impl:       'sdpa' | 'fmha' | 'flex_attention' (default 'sdpa').
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.encoder        = ViTEncoder(cfg)
        self.target_encoder = EMA(ViTEncoder(cfg), decay=cfg.get("ema_decay", 0.996))
        self.predictor      = _Predictor(cfg)
        self.context_ratio  = cfg.get("context_ratio", 0.5)
        self.target_ratio   = cfg.get("target_ratio", 0.25)

    def _sample_mask(self, N: int, ratio: float, device) -> torch.Tensor:
        return torch.randperm(N, device=device)[:max(1, int(N * ratio))]

    def _step(self, batch):
        x = batch["image"]
        ctx_patches = self.encoder(x, return_patches=True)
        with torch.no_grad():
            tgt_patches = self.target_encoder(x, return_patches=True)
        N        = ctx_patches.shape[1]
        ctx_idx  = self._sample_mask(N, self.context_ratio, x.device)
        tgt_idx  = self._sample_mask(N, self.target_ratio,  x.device)
        pred     = self.predictor(ctx_patches[:, ctx_idx], len(tgt_idx))
        return F.mse_loss(pred, tgt_patches[:, tgt_idx].detach())

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self._log_lr()
        self.target_encoder.update(self.encoder)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
