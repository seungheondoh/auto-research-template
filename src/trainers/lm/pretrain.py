"""Causal LM pretraining on raw text (next-token prediction, no prompt masking)."""

from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.lm import build_lm


class PretrainTrainer(BaseTrainer):
    """Standard CLM pretraining.

    Batch keys:
        input_ids [B, L], attention_mask [B, L], labels [B, L]
        (labels == input_ids for next-token prediction; padding positions set to -100)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model, self.tokenizer = build_lm(cfg)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        out = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("train_loss", out.loss, prog_bar=True, on_step=True)
        self._log_lr()
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("val_loss", out.loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return out.loss
