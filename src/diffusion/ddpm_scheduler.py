import torch


class DDPMScheduler:

    num_timesteps: int
    _betas: torch.Tensor
    _alphas: torch.Tensor
    _alpha_bars: torch.Tensor

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02) -> None:
        self.num_timesteps = num_timesteps
        """
        betas: small noise schedules
        """
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self._betas = betas
        self._alphas = alphas
        self._alpha_bars = alpha_bars

    def add_noise(
        self, z0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean image

        Markov chain: z0 -> z1 -> ... -> zt

        Single step:
        zt = √(alpha_t) * z_{t-1} + √(1 - alpha_t) * eps_t
        q(zt | z_{t-1}) = N(zt; √(alpha_t) * z_{t-1}, beta_t * I)

        Closed form:
        zt = √(alpha_bar_t) * z0 + √(1 - alpha_bar_t) * eps
        q(zt | z0) = N(zt; √(alpha_bar_t) * z0, (1 - alpha_bar_t) * I)
        q(z_{t-1} | z0) = N(z_{t-1}; √(alpha_bar_{t-1}) * z0, (1 - alpha_bar_{t-1}) * I)

        where alpha_bar_t = prod(alpha_i) for i=1 to t

        Args:
            z0: clean latent
            t: timestep
            eps: random Gaussian noise eps ~ N(0, I)
        Returns:
            zt: noisy latent at timestep t
        """
        return (
            torch.sqrt(self._alpha_bars.to(device))[t].view(-1, 1, 1, 1) * z0
            + torch.sqrt(1.0 - self._alpha_bars.to(device))[t].view(-1, 1, 1, 1) * eps
        )

    def step(
        self, eps_theta: torch.Tensor, t: torch.Tensor, zt: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse diffusion process: denoise one step (DDPM sampling)

        Given zt and predicted noise eps_theta(zt, t), compute z_{t-1}

        Step 0: Predict z0 from zt
        z0_hat = (zt - √(1 - alpha_bar_t) * eps_theta) / √(alpha_bar_t)

        Bayes' rule
        q(z_{t-1} | zt, z0) = q(zt | z_{t-1}, z0) * q(z_{t-1} | z0) / q(zt | z0)
        1) q(zt | z_{t-1}, z0) = N(zt; √(alpha_t) * z_{t-1}, beta_t * I) ... Single step
        2) q(z_{t-1} | z0) = N(z_{t-1}; √(alpha_bar_{t-1}) * z0, (1 - alpha_bar_{t-1}) * I) ... Closed form
        3) q(zt | z0) = N(zt; √(alpha_bar_t) * z0, (1 - alpha_bar_t) * I) ... Closed form
        q(z_{t-1} | zt, z0) = N(z_{t-1}; μ, σ^2 * I)
        eps_theta noise prediction from UNet -> z0_hat from eps_theta with Closed form
        -> posterior mean(μ) formula approximating z0 with z0_hat
        μ ≈ (zt - √(1 - alpha_t) · eps_theta) / √(alpha_t) ... posterior mean formula
        σ² ≈ beta_t
        z_{t-1} = μ + √(beta_t) * Z,  where Z ~ N(0, I) ... add noise for stochasticity (except at t=0)

        Args:
            eps_theta: predicted noise [B, C, H, W]
            t: current timestep (scalar or [B])
            zt: current noisy latent [B, C, H, W]
        Returns:
            z_t_minus_one_sampled: denoised latent z_{t-1}
        """
        # Handle both scalar and batch timesteps
        if t.dim() == 0:
            t = t.unsqueeze(0)
        device = zt.device
        # Get alpha and beta for timestep t
        alpha = self._alphas.to(device)[t].view(-1, 1, 1, 1)
        beta = self._betas.to(device)[t].view(-1, 1, 1, 1)
        # Compute posterior mean
        posterior_mean = (zt - torch.sqrt(1 - alpha) * eps_theta) / torch.sqrt(alpha)
        Z = torch.randn_like(zt) if t[0].item() > 0 else 0
        z_t_minus_one_sampled = posterior_mean + torch.sqrt(beta) * Z
        return z_t_minus_one_sampled
