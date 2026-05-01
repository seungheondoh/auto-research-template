import datetime
import json
import os
import subprocess

import lightning as L
from lightning.pytorch.callbacks import Callback


def _ascii_chart(values: list, label: str = "", width: int = 50, height: int = 6) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1e-9
    cols = min(len(values), width)
    step = max(1, len(values) // cols)
    sampled = values[::step][:cols]
    rows = []
    for h in range(height, 0, -1):
        threshold = lo + span * (h / height)
        row = "".join("█" if v >= threshold else " " for v in sampled)
        y = f"{threshold:.3f}" if h in (height, 1) else "       "
        lbl = label if h == height else " " * len(label)
        rows.append(f"{lbl} {y} |{row}|")
    rows.append(f"{' ' * (len(label) + 9)} └{'─' * cols}┘")
    rows.append(f"{' ' * (len(label) + 10)}step 1{' ' * max(0, cols - 10)}step {len(values)}")
    return "\n".join(rows)


class ObsidianLogCallback(Callback):
    """
    Writes a live-updating Markdown research note to an Obsidian vault.

    Note path: {OBSIDIAN_VAULT_PATH}/experiments/{experiment_name}.md
    Auto-commits and pushes to git after every validation epoch.
    Set OBSIDIAN_VAULT_PATH env var or pass vault_path explicitly.
    """

    def __init__(self, vault_path: str = None, experiment_name: str = "default",
                 log_every_n_epochs: int = 1):
        super().__init__()
        self.vault_path = vault_path or os.environ.get("OBSIDIAN_VAULT_PATH", "./obsidian_vault")
        self.experiment_name = experiment_name
        self.log_every_n_epochs = log_every_n_epochs
        self._note_path = os.path.join(self.vault_path, "experiments", f"{experiment_name}.md")
        self._metric_rows: list[dict] = []
        self._samples: list[dict] = []
        self._epoch_count = 0
        self._cfg_snapshot: dict = {}
        self._start_time = ""

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        os.makedirs(os.path.dirname(self._note_path), exist_ok=True)
        self._start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg = pl_module.cfg
        self._cfg_snapshot = {
            "trainer_type": cfg.get("trainer_type", "?"),
            "model_type": cfg.get("model_type", cfg.get("model_path", "?")),
            "steps": trainer.max_steps,
            "precision": cfg.get("precision", "?"),
            "lr": cfg.get("lr", "?"),
            "batch_size": cfg.get("batch_size", "?"),
        }
        self._write_note("running")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_count += 1
        if self._epoch_count % self.log_every_n_epochs != 0:
            return
        m = trainer.callback_metrics
        self._metric_rows.append({
            "epoch": self._epoch_count,
            "step": trainer.global_step,
            "train_loss": f"{float(m.get('train_loss', -1)):.4f}",
            "val_loss": f"{float(m.get('val_loss', -1)):.4f}",
            "lr": f"{float(m.get('learning_rate', m.get('lr-AdamW', -1))):.2e}",
        })
        sample = self._load_sample(getattr(pl_module, "expdir", ""), trainer.global_step)
        if sample:
            self._samples.append({"epoch": self._epoch_count, "step": trainer.global_step, "text": sample})
        self._write_note("running")

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._write_note("completed")

    def on_exception(self, trainer: L.Trainer, pl_module: L.LightningModule, exception: Exception) -> None:
        self._write_note("failed", error=str(exception))

    def _write_note(self, status: str, error: str = "") -> None:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best = min((r["val_loss"] for r in self._metric_rows), default="N/A")
        lines = [
            "---",
            f"tags: [ml, experiment, {self._cfg_snapshot.get('trainer_type', 'research')}]",
            f"status: {status}",
            f"experiment: {self.experiment_name}",
            f"started: {self._start_time}",
            f"updated: {now}",
            f"best_val_loss: {best}",
            "---",
            "",
            f"# Experiment: {self.experiment_name}",
            "",
            f"> **{status.upper()}** | Started: `{self._start_time}` | Updated: `{now}`",
            "",
            "## Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
        ] + [f"| `{k}` | `{v}` |" for k, v in self._cfg_snapshot.items()] + [
            "",
            "## Training Metrics",
            "",
            "| Epoch | Step | Train Loss | Val Loss | LR |",
            "|-------|------|-----------|---------|-----|",
        ] + [
            f"| {r['epoch']} | {r['step']} | {r['train_loss']} | {r['val_loss']} | {r['lr']} |"
            for r in self._metric_rows
        ]

        if len(self._metric_rows) > 1:
            val_losses = [float(r["val_loss"]) for r in self._metric_rows]
            train_losses = [float(r["train_loss"]) for r in self._metric_rows]
            lines += [
                "", "## Loss Curve (ASCII)", "", "```",
                _ascii_chart(val_losses, label="val_loss  ", width=50),
                _ascii_chart(train_losses, label="train_loss", width=50),
                "```",
            ]

        if self._samples:
            lines += ["", "## Samples", ""]
            for s in self._samples[-10:]:
                lines += [f"### Epoch {s['epoch']} (step {s['step']})", "", f"> {s['text']}", ""]

        if error:
            lines += ["", "## Error", "", "```", error[:1000], "```"]

        with open(self._note_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self._git_push(status)

    def _git_push(self, status: str) -> None:
        try:
            msg = f"[{self.experiment_name}] {status} — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
            subprocess.run(["git", "-C", self.vault_path, "add", os.path.abspath(self._note_path)],
                           check=True, capture_output=True)
            diff = subprocess.run(["git", "-C", self.vault_path, "diff", "--cached", "--quiet"],
                                  capture_output=True)
            if diff.returncode != 0:
                subprocess.run(["git", "-C", self.vault_path, "commit", "-m", msg],
                               check=True, capture_output=True)
                subprocess.run(["git", "-C", self.vault_path, "push"],
                               check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"[ObsidianCallback] git push failed: {e.stderr.decode()[:200]}")

    def _load_sample(self, expdir: str, step: int) -> str:
        path = os.path.join(expdir, "monitor", f"{step}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("output", "")
        return ""
