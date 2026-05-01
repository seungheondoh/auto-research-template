"""GRPO (Group Relative Policy Optimization) post-training trainer.

Reference: DeepSeek-R1 / Shao et al., "DeepSeekMath" (2024).

Algorithm:
    For each prompt x:
      1. Sample G completions {y_1, …, y_G} from current policy π_θ.
      2. Score each with reward_fn(x, y_i) → r_i.
      3. Compute group-relative advantage: A_i = (r_i - mean(r)) / (std(r) + ε).
      4. REINFORCE with clipped importance ratio (PPO-style clip):
         L = -E[ min(ρ_i * A_i, clip(ρ_i, 1-ε, 1+ε) * A_i) ]
         where ρ_i = π_θ(y_i|x) / π_ref(y_i|x).
      5. Optional KL penalty: + β * KL(π_θ || π_ref).

Batch keys:
    prompt_ids [B, Lp], prompt_mask [B, Lp]   — the prompt
    response_ids [B, G, Lr]                    — G sampled completions per prompt
    response_mask [B, G, Lr]
    rewards [B, G]                             — scalar rewards per completion (optional)

Override `compute_rewards()` to add an online reward signal.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from src.trainers.base_trainer import BaseTrainer
from src.modules.lm import build_lm


def _log_probs(model, input_ids, attention_mask) -> torch.Tensor:
    """Per-token log-probs [B, L-1, V]."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return F.log_softmax(out.logits[:, :-1], dim=-1)


def _response_log_prob(model, prompt_ids, response_ids, response_mask) -> torch.Tensor:
    """Summed log-prob of response tokens given prompt → [B]."""
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    mask = torch.cat([torch.ones_like(prompt_ids), response_mask], dim=1)
    log_probs = _log_probs(model, input_ids, mask)   # [B, L-1, V]
    Lp = prompt_ids.shape[1]
    resp_lp = log_probs[:, Lp - 1:]                  # [B, Lr, V]
    token_lp = resp_lp.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
    return (token_lp * response_mask.float()).sum(dim=-1)


class PosttrainTrainer(BaseTrainer):
    """GRPO post-training over sampled completions.

    Config keys:
        grpo_G:           number of completions per prompt (default 8).
        grpo_clip_eps:    PPO clip epsilon (default 0.2).
        grpo_kl_beta:     KL penalty weight (default 0.01, set 0 to disable).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.model, self.tokenizer = build_lm(cfg)
        self.ref_model, _ = build_lm(cfg, for_reference=True)
        self.ref_model.cpu()
        self.G = cfg.get("grpo_G", 8)
        self.clip_eps = cfg.get("grpo_clip_eps", 0.2)
        self.kl_beta = cfg.get("grpo_kl_beta", 0.01)

    def compute_rewards(self, prompt_ids, response_ids, response_mask) -> torch.Tensor:
        """Override with your reward function. Returns [B, G] float tensor."""
        raise NotImplementedError(
            "Override compute_rewards() with your task-specific reward function. "
            "Expected output: [B, G] float tensor."
        )

    def _grpo_loss(self, batch) -> tuple[torch.Tensor, dict]:
        B = batch["prompt_ids"].shape[0]
        G = batch["response_ids"].shape[1]
        dev = self.device
        ref = self.ref_model.to(dev)

        def flat(t):
            return t.view(B * G, -1)

        prompt_rep = batch["prompt_ids"].unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)
        resp_flat  = flat(batch["response_ids"])
        mask_flat  = flat(batch["response_mask"])

        pi_logp = _response_log_prob(self.model, prompt_rep, resp_flat, mask_flat).view(B, G)
        with torch.no_grad():
            ref_logp = _response_log_prob(ref, prompt_rep, resp_flat, mask_flat).view(B, G)

        self.ref_model.cpu()

        rewards = batch.get("rewards")
        if rewards is None:
            rewards = self.compute_rewards(
                batch["prompt_ids"], batch["response_ids"], batch["response_mask"]
            )

        # Group-relative advantage
        mean_r = rewards.mean(dim=1, keepdim=True)
        std_r  = rewards.std(dim=1, keepdim=True).clamp(min=1e-8)
        A = (rewards - mean_r) / std_r                  # [B, G]

        # Importance ratio + clip
        log_ratio = pi_logp - ref_logp.detach()
        ratio = log_ratio.exp()
        surr1 = ratio * A
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * A
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty
        kl = (ref_logp.detach() - pi_logp).mean()
        loss = policy_loss + self.kl_beta * kl

        return loss, {"reward_mean": rewards.mean(), "kl": kl, "policy_loss": policy_loss}

    def training_step(self, batch, batch_idx):
        loss, aux = self._grpo_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        for k, v in aux.items():
            self.log(f"train_{k}", v, on_step=True)
        self._log_lr()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, aux = self._grpo_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        for k, v in aux.items():
            self.log(f"val_{k}", v, on_epoch=True, sync_dist=True)
        return loss
