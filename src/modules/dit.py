"""DiT and ViT backbone using components from transformers.py.

Shared components pulled from transformers.py:
  - RMSNorm        : replaces LayerNorm in all blocks
  - Attention      : GQA + RoPE + multi-backend (sdpa / fmha / flex_attention)
  - FeedForward    : SwiGLU (w1·silu * w3 → w2)
  - RotaryEmbedding: 1-D RoPE over the flattened patch sequence

DiT     — adaLN-Zero denoiser for generative trainers (diffusion / flow / drift)
ViTEncoder — patch encoder for representation and VAE trainers
ViTDecoder — patch decoder for VAE trainers

Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICLR 2023).
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

from src.modules.transformers import (
    RMSNorm,
    Attention,
    FeedForward,
    RotaryEmbedding,
    BaseTransformerArgs,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN affine transform: x * (1 + scale) + shift, broadcast over sequence."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _make_args(cfg, dim_key: str = "hidden_size") -> BaseTransformerArgs:
    """Build BaseTransformerArgs from an OmegaConf config dict."""
    dim = cfg.get(dim_key, 384)
    n_heads = cfg.get("num_heads", 6)
    return BaseTransformerArgs(
        dim=dim,
        n_layers=cfg.get("depth", 6),
        n_heads=n_heads,
        head_dim=dim // n_heads,
        n_kv_heads=cfg.get("n_kv_heads", n_heads),        # default: full MHA
        ffn_dim_multiplier=cfg.get("ffn_dim_multiplier", None),
        multiple_of=cfg.get("multiple_of", 256),
        norm_eps=cfg.get("norm_eps", 1e-5),
        rope_theta=cfg.get("rope_theta", 10000.0),
        max_seqlen=cfg.get("max_seqlen", 1024),
    )


# ── patch utilities ───────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), "b d h w -> b (h w) d")


class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = t[:, None].float() * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# ── DiT block (adaLN-Zero) ────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    """adaLN-Zero block using GQA Attention + SwiGLU FFN from transformers.py.

    Conditioning vector c (timestep + optional class) modulates both the
    attention and FFN sub-layers via shift / scale / gate (6 parameters total).
    Gates are zero-initialised so the residual stream starts as identity.
    """

    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        head_dim  = args.head_dim or args.dim // args.n_heads
        n_heads   = args.n_heads  or args.dim // args.head_dim
        n_kv_heads = args.n_kv_heads or n_heads

        self.norm1 = RMSNorm(args.dim, args.norm_eps)
        self.norm2 = RMSNorm(args.dim, args.norm_eps)
        self.attn  = Attention(
            dim=args.dim, head_dim=head_dim,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.ff = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        # adaLN-Zero: zero-init so gates start at 0 → identity residual
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(args.dim, 6 * args.dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                freq_cis: torch.Tensor, attn_impl: str = "sdpa") -> torch.Tensor:
        s_a, sc_a, g_a, s_m, sc_m, g_m = self.adaLN(c).chunk(6, dim=-1)
        h = self.attn(modulate(self.norm1(x), s_a, sc_a), freq_cis, attn_impl=attn_impl)
        x = x + g_a.unsqueeze(1) * h
        x = x + g_m.unsqueeze(1) * self.ff(modulate(self.norm2(x), s_m, sc_m))
        return x


# ── ViT block (no conditioning) ───────────────────────────────────────────────

class SelfAttnBlock(nn.Module):
    """Standard ViT block using GQA Attention + SwiGLU FFN from transformers.py."""

    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        head_dim   = args.head_dim or args.dim // args.n_heads
        n_heads    = args.n_heads  or args.dim // args.head_dim
        n_kv_heads = args.n_kv_heads or n_heads

        self.norm1 = RMSNorm(args.dim, args.norm_eps)
        self.norm2 = RMSNorm(args.dim, args.norm_eps)
        self.attn  = Attention(
            dim=args.dim, head_dim=head_dim,
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.ff = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor,
                attn_impl: str = "sdpa") -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freq_cis, attn_impl=attn_impl)
        x = x + self.ff(self.norm2(x))
        return x


# ── FinalLayer ────────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """adaLN + zero-initialised linear to patch pixels (DiT output head)."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm   = RMSNorm(hidden_size, norm_eps)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN  = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ── DiT denoiser ──────────────────────────────────────────────────────────────

class DiT(nn.Module):
    """Diffusion Transformer — denoiser / velocity predictor.

    Used by: generative/diffusion.py, generative/flow_matching.py, generative/drift.py.

    Config keys (passed as OmegaConf cfg or forwarded via _make_args):
        img_size, patch_size, in_channels
        hidden_size, depth, num_heads, n_kv_heads (default = num_heads → full MHA)
        ffn_dim_multiplier, multiple_of, norm_eps, rope_theta
        num_classes: 0 = unconditional, >0 = class-conditional
        attn_impl: 'sdpa' (default) | 'fmha' | 'flex_attention'
    """

    def __init__(self, cfg):
        super().__init__()
        args = _make_args(cfg)
        self.args       = args
        self.patch_size = cfg.get("patch_size", 4)
        self.in_channels = cfg.get("in_channels", 3)
        img_size        = cfg.get("img_size", 32)
        self.h          = img_size // self.patch_size
        num_patches     = self.h ** 2
        self.attn_impl  = cfg.get("attn_impl", "sdpa")

        self.patch_embed  = PatchEmbed(img_size, self.patch_size, self.in_channels, args.dim)
        self.t_embed      = TimestepEmbed(args.dim)
        self.label_embed  = (
            nn.Embedding(cfg.get("num_classes") + 1, args.dim)
            if cfg.get("num_classes", 0) > 0 else None
        )
        self.rope         = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=num_patches,
        )
        self.blocks       = nn.ModuleList([DiTBlock(args) for _ in range(args.n_layers)])
        self.final_layer  = FinalLayer(args.dim, self.patch_size, self.in_channels, args.norm_eps)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, h = self.patch_size, self.h
        return rearrange(x, "b (h w) (p q c) -> b c (h p) (w q)", h=h, w=h, p=p, q=p)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: noisy image [B, C, H, W].
            t: timestep [B] — int for DDPM, float in [0,1] for flow/drift.
            y: optional class labels [B].
        Returns:
            predicted noise / velocity [B, C, H, W].
        """
        freq_cis = self.rope(seqlen=x.shape[2] // self.patch_size * x.shape[3] // self.patch_size)
        x = self.patch_embed(x)
        c = self.t_embed(t.long() if t.is_floating_point() else t)
        if self.label_embed is not None and y is not None:
            c = c + self.label_embed(y)
        for block in self.blocks:
            x = block(x, c, freq_cis, self.attn_impl)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


# ── ViT encoder (representation + VAE) ───────────────────────────────────────

class ViTEncoder(nn.Module):
    """ViT encoder: image → CLS token [B, D] or patch tokens [B, N, D].

    Used by: representation/infonce.py, representation/jepa.py, vae/vae.py, vae/vqvae.py.

    Config keys: img_size, patch_size, in_channels, hidden_size, depth, num_heads,
                 n_kv_heads, norm_eps, rope_theta, attn_impl.
    """

    def __init__(self, cfg):
        super().__init__()
        args = _make_args(cfg)
        self.args      = args
        patch_size     = cfg.get("patch_size", 4)
        img_size       = cfg.get("img_size", 32)
        in_channels    = cfg.get("in_channels", 3)
        num_patches    = (img_size // patch_size) ** 2
        self.attn_impl = cfg.get("attn_impl", "sdpa")

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, args.dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, args.dim))
        self.rope        = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=num_patches + 1,   # +1 for CLS
        )
        self.blocks = nn.ModuleList([SelfAttnBlock(args) for _ in range(args.n_layers)])
        self.norm   = RMSNorm(args.dim, args.norm_eps)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        """
        Args:
            x:              [B, C, H, W].
            return_patches: True → patch tokens [B, N, D]; False → CLS [B, D].
        """
        tokens   = self.patch_embed(x)
        cls      = self.cls_token.expand(x.shape[0], -1, -1)
        tokens   = torch.cat([cls, tokens], dim=1)              # [B, N+1, D]
        freq_cis = self.rope(seqlen=tokens.shape[1])
        for block in self.blocks:
            tokens = block(tokens, freq_cis, self.attn_impl)
        tokens = self.norm(tokens)
        return tokens[:, 1:] if return_patches else tokens[:, 0]


# ── ViT decoder (VAE) ─────────────────────────────────────────────────────────

class ViTDecoder(nn.Module):
    """ViT decoder: latent [B, latent_dim] or patch tokens [B, N, D] → image [B, C, H, W].

    Used by: vae/vae.py (continuous latent), vae/vqvae.py (quantized patches).
    Pass `from_patches=True` when z is already [B, N, D] (VQ-VAE path).

    Config keys: img_size, patch_size, out_channels (= in_channels), hidden_size,
                 dec_depth (default 4), num_heads, n_kv_heads, norm_eps, rope_theta,
                 latent_dim, attn_impl.
    """

    def __init__(self, cfg):
        super().__init__()
        # decoder may use a shallower transformer
        dec_cfg = dict(cfg)
        dec_cfg["depth"] = cfg.get("dec_depth", 4)
        args = _make_args(dec_cfg)
        self.args = args

        patch_size   = cfg.get("patch_size", 4)
        img_size     = cfg.get("img_size", 32)
        out_channels = cfg.get("in_channels", 3)
        latent_dim   = cfg.get("latent_dim", 256)
        num_patches  = (img_size // patch_size) ** 2
        self.h           = img_size // patch_size
        self.patch_size  = patch_size
        self.out_channels = out_channels
        self.num_patches = num_patches
        self.attn_impl   = cfg.get("attn_impl", "sdpa")

        self.latent_proj = nn.Linear(latent_dim, num_patches * args.dim)  # continuous VAE path
        self.patch_proj  = nn.Linear(args.dim, args.dim)                   # VQ-VAE patch path
        self.rope   = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=num_patches,
        )
        self.blocks = nn.ModuleList([SelfAttnBlock(args) for _ in range(args.n_layers)])
        self.norm   = RMSNorm(args.dim, args.norm_eps)
        self.final  = nn.Linear(args.dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, h = self.patch_size, self.h
        return rearrange(x, "b (h w) (p q c) -> b c (h p) (w q)", h=h, w=h, p=p, q=p)

    def forward(self, z: torch.Tensor, from_patches: bool = False) -> torch.Tensor:
        """
        Args:
            z:            [B, latent_dim] for VAE, or [B, N, D] for VQ-VAE.
            from_patches: True when z is already patch-shaped [B, N, D].
        """
        if from_patches:
            x = self.patch_proj(z)
        else:
            x = self.latent_proj(z).view(z.shape[0], self.num_patches, self.args.dim)
        freq_cis = self.rope(seqlen=self.num_patches)
        for block in self.blocks:
            x = block(x, freq_cis, self.attn_impl)
        x = self.norm(x)
        return self.unpatchify(self.final(x)).tanh()
