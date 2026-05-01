"""
Generic training entry point.

Usage:
    python train.py --config config/lm/default.yaml
    python train.py --config config/generative/ddpm.yaml --gpu 0
    python train.py --config config/representation/contrastive.yaml --resume ./exp/.../last.ckpt
"""
import argparse
import os

import torch
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import OmegaConf

from src.utils.config_utils import instantiate_from_config, save_config
from src.callbacks.slack_callback import SlackNotificationCallback
from src.callbacks.obsidian_callback import ObsidianLogCallback

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args: argparse.Namespace) -> None:
    config = OmegaConf.load(args.config)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["NCCL_P2P_DISABLE"] = "1"

    expdir = os.path.join(config.lightning.logdir, config.project, config.lightning.version_name)
    os.makedirs(expdir, exist_ok=True)
    save_config(config, os.path.join(expdir, "config.yaml"))

    L.seed_everything(42)

    data_module = instantiate_from_config(config.data)
    data_module.prepare_data()

    model = instantiate_from_config(config.model)

    # ── Trainer strategy ──────────────────────────────────────────────────────
    trainer_cfg = dict(config.lightning.trainer)
    strategy_name = trainer_cfg.pop("strategy", "auto")
    strategy = DDPStrategy(find_unused_parameters=True) if strategy_name == "ddp" else strategy_name

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=config.lightning.logdir,
        name=config.project,
        version=config.lightning.version_name,
    )

    # ── Callbacks from config ─────────────────────────────────────────────────
    callbacks = []
    for cb_conf in config.lightning.get("callbacks", {}).values():
        callbacks.append(instantiate_from_config(cb_conf))

    # ── Slack (env var opt-in) ────────────────────────────────────────────────
    if os.environ.get("SLACK_WEBHOOK_URL"):
        callbacks.append(SlackNotificationCallback(
            experiment_name=config.lightning.version_name,
            notify_every_n_epochs=config.lightning.get("slack_notify_every_n_epochs", 1),
        ))
        print(f"[train] Slack → {os.environ['SLACK_WEBHOOK_URL'][:40]}...")

    # ── Obsidian (always on, defaults to ./obsidian_vault) ───────────────────
    callbacks.append(ObsidianLogCallback(
        experiment_name=config.lightning.version_name,
        log_every_n_epochs=config.lightning.get("obsidian_log_every_n_epochs", 1),
    ))
    vault = os.environ.get("OBSIDIAN_VAULT_PATH", "./obsidian_vault")
    print(f"[train] Obsidian → {vault}/experiments/{config.lightning.version_name}.md")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        callbacks=callbacks,
        strategy=strategy,
        devices="auto",
        logger=logger,
        **trainer_cfg,
    )
    trainer.fit(model, data_module, ckpt_path=args.resume or None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    main(parser.parse_args())
