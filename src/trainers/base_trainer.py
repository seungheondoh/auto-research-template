import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.utils.scheduler import LinearWarmupCosineDecay


class BaseTrainer(L.LightningModule):
    """Common optimizer / scheduler wiring shared by all trainer types."""

    def __init__(self, cfg):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.expdir = cfg.get("expdir", "./exp")

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.cfg.lr,
            weight_decay=self.cfg.get("weight_decay", 0.01),
            betas=tuple(self.cfg.get("betas", [0.9, 0.99])),
        )
        scheduler = LinearWarmupCosineDecay(
            optimizer,
            warmup_steps=self.cfg.get("warmup_steps", 100),
            total_steps=self.cfg.steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _log_lr(self):
        opt = self.optimizers()
        if opt is not None:
            self.log("learning_rate", opt.param_groups[0]["lr"], on_step=True)
