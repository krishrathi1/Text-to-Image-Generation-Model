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
        self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean image

        Markov chain: x0 -> x1 -> ... -> xt

        Single step:
        xt = √(alpha_t) * x_{t-1} + √(1 - alpha_t) * eps_t
        q(xt | x_{t-1}) = N(xt; √(alpha_t) * x_{t-1}, beta_t * I)

        Closed form:
        xt = √(alpha_bar_t) * x0 + √(1 - alpha_bar_t) * eps
        q(xt | x0) = N(xt; √(alpha_bar_t) * x0, (1 - alpha_bar_t) * I)
        q(x_{t-1} | x0) = N(x_{t-1}; √(alpha_bar_{t-1}) * x0, (1 - alpha_bar_{t-1}) * I)

        where alpha_bar_t = prod(alpha_i) for i=1 to t

        Args:
            x0: clean image
            t: timestep
            eps: random Gaussian noise eps ~ N(0, I)
        Returns:
            xt: noisy image at timestep t
        """
        return (
            torch.sqrt(self._alpha_bars.to(device))[t].view(-1, 1, 1, 1) * x0
            + torch.sqrt(1.0 - self._alpha_bars.to(device))[t].view(-1, 1, 1, 1) * eps
        )

    def step(
        self, eps_theta: torch.Tensor, t: torch.Tensor, xt: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse diffusion process: denoise one step (DDPM sampling)

        Given xt and predicted noise eps_theta(xt, t), compute x_{t-1}

        Step 0: Predict x0 from xt
        x0_hat = (xt - √(1 - alpha_bar_t) * eps_theta) / √(alpha_bar_t)

        Bayes' rule
        q(x_{t-1} | xt, x0) = q(xt | x_{t-1}, x0) * q(x_{t-1} | x0) / q(xt | x0)
        1) q(xt | x_{t-1}, x0) = N(xt; √(alpha_t) * x_{t-1}, beta_t * I) ... Single step
        2) q(x_{t-1} | x0) = N(x_{t-1}; √(alpha_bar_{t-1}) * x0, (1 - alpha_bar_{t-1}) * I) ... Closed form
        3) q(xt | x0) = N(xt; √(alpha_bar_t) * x0, (1 - alpha_bar_t) * I) ... Closed form
        q(x_{t-1} | xt, x0) = N(x_{t-1}; μ, σ^2 * I)
        eps_theta noise prediction from UNet -> x0_hat from eps_theta with Closed form
        -> posterior mean(μ) formula approximating x0 with x0_hat
        μ ≈ (xt - √(1 - alpha_t) · eps_theta) / √(alpha_t) ... posterior mean formula
        σ² ≈ beta_t
        x_{t-1} = μ + √(beta_t) * z,  where z ~ N(0, I) ... add noise for stochasticity (except at t=0)

        Args:
            eps_theta: predicted noise [B, C, H, W]
            t: current timestep (scalar or [B])
            xt: current noisy image [B, C, H, W]
        Returns:
            x_t_minus_one_sampled: denoised image x_{t-1}
        """
        # Handle both scalar and batch timesteps
        if t.dim() == 0:
            t = t.unsqueeze(0)
        device = xt.device
        # Get alpha and beta for timestep t
        alpha = self._alphas.to(device)[t].view(-1, 1, 1, 1)
        beta = self._betas.to(device)[t].view(-1, 1, 1, 1)
        # Compute posterior mean
        posterior_mean = (xt - torch.sqrt(1 - alpha) * eps_theta) / torch.sqrt(alpha)
        z = torch.randn_like(xt) if t[0].item() > 0 else 0
        x_t_minus_one_sampled = posterior_mean + torch.sqrt(beta) * z
        return x_t_minus_one_sampled
