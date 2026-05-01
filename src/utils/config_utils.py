import importlib

import torch
from omegaconf import DictConfig, OmegaConf


def get_obj_from_str(string: str):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config: DictConfig, **kwargs):
    """Instantiate any class from a config with a `target` key and optional `params`."""
    target = config["target"]
    params = dict(config.get("params", {}))
    params.update(kwargs)
    return get_obj_from_str(target)(**params)


def save_config(cfg: DictConfig, path: str) -> None:
    OmegaConf.save(cfg, path)


def load_config(path: str) -> DictConfig:
    return OmegaConf.load(path)


def load_ckpt_state_dict(ckpt_path: str) -> dict:
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(ckpt_path)
    return torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
