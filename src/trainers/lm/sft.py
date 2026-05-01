"""Supervised Fine-Tuning (SFT) with instruction formatting and prompt loss masking."""

import torch
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.lm import build_lm


class SFTTrainer(BaseTrainer):
    """Instruction-following SFT with LoRA.

    Batch keys:
        input_ids [B, L]       — full prompt + response token ids
        attention_mask [B, L]
        labels [B, L]          — prompt positions set to -100 (masked from loss),
                                 response positions = token ids

    The datamodule is responsible for constructing labels with the prompt mask.
    A typical pattern:

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100   # ignore prompt tokens in CE loss
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

    # ------------------------------------------------------------------
    # generation helper (for qualitative eval in callbacks)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.cfg.get("gen_temperature", 0.7),
            top_p=self.cfg.get("gen_top_p", 0.9),
        )
        return self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
