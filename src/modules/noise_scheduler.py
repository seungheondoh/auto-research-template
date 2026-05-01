import torch


class DDPMScheduler:
    """DDPM cosine / linear noise schedule.

    Buffers are registered on first `.to(device)` call via the trainer.
    Holds: betas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod.
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = num_timesteps
        betas = self._make_betas(schedule, num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        self.register = {
            "betas": betas,
            "sqrt_acp": acp.sqrt(),
            "sqrt_one_minus_acp": (1.0 - acp).sqrt(),
            "acp": acp,
        }

    @staticmethod
    def _make_betas(schedule: str, T: int, b0: float, bT: float) -> torch.Tensor:
        if schedule == "linear":
            return torch.linspace(b0, bT, T)
        if schedule == "cosine":
            t = torch.linspace(0, T, T + 1)
            f = torch.cos(((t / T) + 0.008) / 1.008 * torch.pi / 2) ** 2
            f = f / f[0]
            return (1 - f[1:] / f[:-1]).clamp(1e-4, 0.9999)
        raise ValueError(f"Unknown schedule '{schedule}'")

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor, device) -> torch.Tensor:
        """Forward process: q(x_t | x_0) = sqrt(acp_t)*x0 + sqrt(1-acp_t)*noise."""
        s_a = self.register["sqrt_acp"].to(device)[t].view(-1, 1, 1, 1)
        s_b = self.register["sqrt_one_minus_acp"].to(device)[t].view(-1, 1, 1, 1)
        return s_a * x0 + s_b * noise

    def get_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
                   prediction_type: str, device) -> torch.Tensor:
        s_a = self.register["sqrt_acp"].to(device)[t].view(-1, 1, 1, 1)
        s_b = self.register["sqrt_one_minus_acp"].to(device)[t].view(-1, 1, 1, 1)
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "x_start":
            return x0
        if prediction_type == "v_prediction":
            return s_a * noise - s_b * x0
        raise ValueError(f"Unknown prediction_type '{prediction_type}'")


def get_flow_interpolation(x0: torch.Tensor, t: torch.Tensor,
                           alpha_fn=None, sigma_fn=None):
    """Stochastic interpolant: x_t = alpha(t)*x0 + sigma(t)*noise.

    Default (linear): alpha(t) = 1-t, sigma(t) = t  →  OT flow matching.
    Override alpha_fn / sigma_fn for other interpolants (e.g., VP, sub-VP).

    Returns:
        xt:            interpolated sample
        drift_target:  d/dt x_t = alpha'(t)*x0 + sigma'(t)*noise
        noise:         the noise sample used
    """
    noise = torch.randn_like(x0)
    t_b = t.view(-1, 1, 1, 1)
    if alpha_fn is None:
        alpha, d_alpha = 1.0 - t_b, -1.0
    else:
        alpha, d_alpha = alpha_fn(t_b)
    if sigma_fn is None:
        sigma, d_sigma = t_b, 1.0
    else:
        sigma, d_sigma = sigma_fn(t_b)

    xt = alpha * x0 + sigma * noise
    drift = d_alpha * x0 + d_sigma * noise
    return xt, drift, noise
