"""β-VAE with ViTEncoder + ViTDecoder (DiT-based).

Loss: L = ||x - x_hat||^2 + β * KL(q(z|x) || N(0, I))

Batch keys:
    image [B, C, H, W]
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.dit import ViTEncoder, ViTDecoder
from src.modules.losses import elbo_loss


class VAETrainer(BaseTrainer):
    """β-VAE trainer.

    Config keys:
        img_size, patch_size, in_channels, hidden_size, depth, num_heads — ViT backbone
        dec_depth:  decoder transformer depth (default 4).
        latent_dim: latent space dimension (default 256).
        beta:       KL weight (default 1.0; >1 encourages disentanglement).
        attn_impl:  'sdpa' | 'fmha' | 'flex_attention' (default 'sdpa').
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        hidden     = cfg.get("hidden_size", 384)
        latent_dim = cfg.get("latent_dim", 256)

        self.encoder = ViTEncoder(cfg)
        self.fc_mu   = nn.Linear(hidden, latent_dim)
        self.fc_lv   = nn.Linear(hidden, latent_dim)
        self.decoder = ViTDecoder(cfg)
        self.beta    = cfg.get("beta", 1.0)

    def encode(self, x):
        h  = self.encoder(x)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterise(self, mu, lv):
        return mu + (0.5 * lv).exp() * torch.randn_like(mu)

    def forward(self, x):
        mu, lv = self.encode(x)
        return self.decoder(self.reparameterise(mu, lv)), mu, lv

    def _step(self, batch):
        x = batch["image"]
        x_hat, mu, lv = self(x)
        return elbo_loss(x, x_hat, mu, lv, self.beta)

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
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        for k, v in aux.items():
            self.log(f"val_{k}", v, on_epoch=True, sync_dist=True)
        return loss
