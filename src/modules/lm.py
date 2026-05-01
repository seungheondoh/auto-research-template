"""Qwen LM builder shared by all LM trainers (pretrain / sft / posttrain / rl)."""

from __future__ import annotations
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def build_lm(cfg: DictConfig, *, for_reference: bool = False):
    """Load Qwen and optionally wrap with LoRA.

    Args:
        cfg:           trainer config (model_path, apply_lora, lora_rank, …).
        for_reference: if True, freeze all parameters (DPO reference model).

    Returns:
        (model, tokenizer)
    """
    dtype = torch.bfloat16 if "bf16" in cfg.get("precision", "") else torch.float32
    model_path = cfg.get("model_path", DEFAULT_MODEL)
    cache_dir = cfg.get("cache_dir", "./cache")

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=cache_dir, torch_dtype=dtype
    )

    if for_reference:
        for p in model.parameters():
            p.requires_grad_(False)
        return model, tokenizer

    if cfg.get("apply_lora", True):
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_rank", 64),
            lora_alpha=cfg.get("lora_alpha", 128),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if cfg.get("activation_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model, tokenizer
