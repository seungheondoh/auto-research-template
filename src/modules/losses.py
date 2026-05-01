import torch
import torch.nn.functional as F


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent / InfoNCE on L2-normalised projections.

    Args:
        z1, z2: [B, D] normalised embeddings of two views.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)          # [2N, D]
    sim = (z @ z.T) / temperature           # [2N, 2N]
    sim.fill_diagonal_(float("-inf"))
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(N, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def elbo_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor,
              logvar: torch.Tensor, beta: float = 1.0) -> tuple[torch.Tensor, dict]:
    """ELBO = reconstruction + β·KL.

    Args:
        x, x_hat: [B, C, H, W] original and reconstruction.
        mu, logvar: [B, latent_dim] posterior parameters.
        beta: weight on KL term (β-VAE).
    """
    recon = F.mse_loss(x_hat, x)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl, {"recon": recon, "kl": kl}


def vq_loss(z_e: torch.Tensor, z_q: torch.Tensor, commitment_weight: float = 0.25
            ) -> tuple[torch.Tensor, dict]:
    """VQ-VAE loss: codebook + commitment (straight-through in z_q).

    Args:
        z_e: encoder output [B, N, D].
        z_q: quantized (detached gradient for codebook part) [B, N, D].
    """
    codebook = F.mse_loss(z_q, z_e.detach())
    commitment = F.mse_loss(z_e, z_q.detach())
    return codebook + commitment_weight * commitment, {
        "codebook": codebook, "commitment": commitment
    }
