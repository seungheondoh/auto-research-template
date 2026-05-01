import datetime
import json
import os

import requests
import lightning as L
from lightning.pytorch.callbacks import Callback


def _post(webhook_url: str, payload: dict) -> None:
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[SlackCallback] send failed: {e}")


class SlackNotificationCallback(Callback):
    """
    Posts training lifecycle events to Slack via Incoming Webhook.

    Events: train_start, validation_epoch_end (with generated sample), train_end, exception.
    Set SLACK_WEBHOOK_URL env var or pass webhook_url explicitly.
    """

    def __init__(self, webhook_url: str = None, experiment_name: str = "default",
                 notify_every_n_epochs: int = 1):
        super().__init__()
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
        self.experiment_name = experiment_name
        self.notify_every_n_epochs = notify_every_n_epochs
        self._epoch_count = 0
        self._start_time = None
        self._best_val_loss = float("inf")

    def _send(self, blocks: list) -> None:
        if not self.webhook_url:
            print("[SlackCallback] No webhook URL set — skipping.")
            return
        _post(self.webhook_url, {"blocks": blocks})

    def _header(self, emoji: str, text: str) -> dict:
        return {"type": "header", "text": {"type": "plain_text", "text": f"{emoji} {text}"}}

    def _section(self, text: str) -> dict:
        return {"type": "section", "text": {"type": "mrkdwn", "text": text}}

    def _divider(self) -> dict:
        return {"type": "divider"}

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start_time = datetime.datetime.now()
        cfg = pl_module.cfg
        self._send([
            self._header("rocket", f"Training Started — {self.experiment_name}"),
            self._section(
                f"*Trainer type:* `{cfg.get('trainer_type', '?')}`\n"
                f"*Model type:* `{cfg.get('model_type', cfg.get('model_path', '?'))}`\n"
                f"*Steps:* `{trainer.max_steps}`  |  "
                f"*Precision:* `{cfg.get('precision', '?')}`\n"
                f"*Started:* `{self._start_time:%Y-%m-%d %H:%M:%S}`"
            ),
        ])

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_count += 1
        if self._epoch_count % self.notify_every_n_epochs != 0:
            return
        metrics = trainer.callback_metrics
        val_loss = float(metrics.get("val_loss", -1))
        train_loss = float(metrics.get("train_loss", -1))
        lr = float(metrics.get("learning_rate", metrics.get("lr-AdamW", -1)))
        improved = val_loss < self._best_val_loss
        if improved:
            self._best_val_loss = val_loss
        emoji = "white_check_mark" if improved else "chart_with_upwards_trend"
        blocks = [
            self._header(emoji, f"Epoch {self._epoch_count} — {self.experiment_name}"),
            self._section(
                f"*Step:* `{trainer.global_step}`\n"
                f"*Val Loss:* `{val_loss:.4f}` {'← best :star:' if improved else ''}\n"
                f"*Train Loss:* `{train_loss:.4f}`\n"
                f"*LR:* `{lr:.2e}`"
            ),
        ]
        sample = self._load_sample(getattr(pl_module, "expdir", ""), trainer.global_step)
        if sample:
            blocks += [self._divider(), self._section(f"*Sample:*\n```{sample[:500]}```")]
        self._send(blocks)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        elapsed = str(datetime.datetime.now() - self._start_time).split(".")[0] if self._start_time else "?"
        self._send([
            self._header("tada", f"Training Complete — {self.experiment_name}"),
            self._section(
                f"*Best Val Loss:* `{self._best_val_loss:.4f}`\n"
                f"*Total Steps:* `{trainer.global_step}`\n"
                f"*Duration:* `{elapsed}`"
            ),
        ])

    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: Exception) -> None:
        self._send([
            self._header("x", f"Error — {self.experiment_name}"),
            self._section(f"```{type(exception).__name__}: {str(exception)[:800]}```"),
        ])

    def _load_sample(self, expdir: str, step: int) -> str:
        path = os.path.join(expdir, "monitor", f"{step}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("output", "")
        return ""
