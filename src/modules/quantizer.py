import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """VQ-VAE vector quantizer with straight-through estimator.

    Args:
        num_embeddings: codebook size K.
        embedding_dim:  dimension D of each code.
        commitment_weight: weight on commitment loss (β in VQ-VAE paper).
    """

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 256,
                 commitment_weight: float = 0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, N, D] encoder output (flat spatial or sequence tokens).

        Returns:
            z_q:    quantized [B, N, D] (straight-through gradient to z).
            loss:   scalar VQ + commitment loss.
            indices:[B, N] codebook indices.
        """
        flat = z.reshape(-1, self.D)                         # [B*N, D]
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.weight.T
            + self.embedding.weight.pow(2).sum(1)
        )                                                    # [B*N, K]
        indices = dist.argmin(dim=1)                         # [B*N]
        z_q_flat = self.embedding(indices)                   # [B*N, D]
        z_q = z_q_flat.view_as(z)

        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        loss = codebook_loss + self.commitment_weight * commitment_loss

        z_q = z + (z_q - z).detach()                        # straight-through
        return z_q, loss, indices.view(z.shape[0], -1)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [B, N] → [B, N, D]."""
        return self.embedding(indices)
