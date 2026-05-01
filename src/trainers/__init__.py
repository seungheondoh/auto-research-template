from src.trainers.lm.pretrain import PretrainTrainer
from src.trainers.lm.sft import SFTTrainer
from src.trainers.lm.posttrain import PosttrainTrainer
from src.trainers.lm.rl import RLTrainer

from src.trainers.representation.infonce import InfoNCETrainer
from src.trainers.representation.jepa import JEPATrainer

from src.trainers.generative.diffusion import DiffusionTrainer
from src.trainers.generative.flow_matching import FlowMatchingTrainer
from src.trainers.generative.drift import DriftTrainer

from src.trainers.vae.vae import VAETrainer
from src.trainers.vae.vqvae import VQVAETrainer


_REGISTRY = {
    # LM
    "pretrain":    PretrainTrainer,
    "sft":         SFTTrainer,
    "posttrain":   PosttrainTrainer,
    "rl":          RLTrainer,
    # Representation
    "infonce":     InfoNCETrainer,
    "jepa":        JEPATrainer,
    # Generative
    "diffusion":   DiffusionTrainer,
    "flow_matching": FlowMatchingTrainer,
    "drift":       DriftTrainer,
    # VAE
    "vae":         VAETrainer,
    "vqvae":       VQVAETrainer,
}


def get_trainer(cfg):
    trainer_type = cfg.get("trainer_type")
    if trainer_type not in _REGISTRY:
        raise ValueError(
            f"Unknown trainer_type '{trainer_type}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[trainer_type](cfg)
