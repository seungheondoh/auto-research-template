"""VQ-VAE with ViTEncoder + VectorQuantizer + ViTDecoder (DiT-based).

Reference: van den Oord et al., "Neural Discrete Representation Learning"
           (NeurIPS 2017).
           Chang et al., "MaskGIT: Masked Generative Image Transformer"
           (CVPR 2022) — uses VQ-VAE tokenizer.

Architecture:
    Encoder: ViTEncoder (return_patches=True) → [B, N, D] patch tokens
    Quantizer: VectorQuantizer → z_q [B, N, D] + indices [B, N]
    Decoder: ViTDecoder (from_patches=True, z_q as input) → x_hat [B, C, H, W]

Loss:
    L = ||x - x_hat||^2 + codebook_loss + β * commitment_loss

The straight-through estimator in VectorQuantizer allows gradients to flow
from the decoder through the quantized tokens to the encoder.

Batch keys:
    image [B, C, H, W]
"""

import torch
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.dit import ViTEncoder, ViTDecoder
from src.modules.quantizer import VectorQuantizer

import torch.nn.functional as F


class VQVAETrainer(BaseTrainer):
    """VQ-VAE trainer.

    Config keys:
        img_size, patch_size, in_channels  — image properties
        hidden_size, depth, num_heads      — ViT backbone
        num_embeddings:   codebook size K (default 512).
        commitment_weight: β for commitment loss (default 0.25).
        dec_depth:         decoder depth (default 4).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        hidden = cfg.get("hidden_size", 384)

        self.encoder = ViTEncoder(
            img_size=cfg.get("img_size", 32),
            patch_size=cfg.get("patch_size", 4),
            in_channels=cfg.get("in_channels", 3),
            hidden_size=hidden,
            depth=cfg.get("depth", 6),
            num_heads=cfg.get("num_heads", 6),
        )
        self.quantizer = VectorQuantizer(
            num_embeddings=cfg.get("num_embeddings", 512),
            embedding_dim=hidden,
            commitment_weight=cfg.get("commitment_weight", 0.25),
        )
        self.decoder = ViTDecoder(
            img_size=cfg.get("img_size", 32),
            patch_size=cfg.get("patch_size", 4),
            out_channels=cfg.get("in_channels", 3),
            hidden_size=hidden,
            depth=cfg.get("dec_depth", 4),
            num_heads=cfg.get("num_heads", 6),
            latent_dim=hidden,
        )

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize image to discrete codebook indices [B, N]."""
        z_e = self.encoder(x, return_patches=True)
        _, _, indices = self.quantizer(z_e)
        return indices

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from codebook indices [B, N] → [B, C, H, W]."""
        z_q = self.quantizer.decode_indices(indices)
        return self.decoder(z_q, from_patches=True)

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x, return_patches=True)    # [B, N, D]
        z_q, vq_loss, indices = self.quantizer(z_e)   # straight-through z_q
        x_hat = self.decoder(z_q, from_patches=True)  # [B, C, H, W]
        return x_hat, vq_loss, indices

    def _step(self, batch):
        x = batch["image"]
        x_hat, vq_loss, _ = self(x)
        recon = F.mse_loss(x_hat, x)
        loss = recon + vq_loss
        return loss, {"recon": recon, "vq": vq_loss}

    def training_step(self, batch, batch_idx):
        loss, aux = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        for k, v in aux.items():
            self.log(f"train_{k}", v, on_step=True)
        self._log_lr()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, aux = self._step(batch)
        self.log("val_loss",  loss, prog_bar=True, on_epoch=True, sync_dist=True)
        for k, v in aux.items():
            self.log(f"val_{k}", v, on_epoch=True, sync_dist=True)
        return loss
