import torch
import torch.nn.functional as F

from .stable_diffusion import StableDiffusion


class StableDiffusionTrainer:
    """
    Training Strategy:
    - CLIP: Pre-trained, frozen (text encoding)
    - VAE: Pre-trained, frozen (image compression)
    - UNet: Train from scratch (noise prediction)
    """

    model: StableDiffusion
    _optimizer: torch.optim.Optimizer
    _device: torch.device
    _max_grad_norm: float

    def __init__(
        self,
        model: StableDiffusion,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_grad_norm=1.0,
    ) -> None:
        self.model = model
        self._optimizer = optimizer
        self._device = device
        self._max_grad_norm = max_grad_norm
        for param in model.vae.parameters():
            param.requires_grad = False
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        for param in model.unet.parameters():
            param.requires_grad = True

    def train_step(self, images: torch.Tensor, text_prompts: list[str]) -> float:
        """
        Single training step for the UNet

        Training objective (DDPM):
        L = E_{x0, eps, t} [||eps - eps_theta(xt, t, c)||^2]

        where:
        - x0: clean image (from VAE encoding)
        - eps: random noise ~ N(0, I)
        - t: random timestep
        - c: text condition (from CLIP)
        - xt: noisy image at timestep t
        - eps_theta: UNet prediction

        Process:
        1. Encode images to latent space using VAE (frozen)
        2. Encode text prompts using CLIP (frozen)
        3. Sample random timesteps t
        4. Add noise to latents: xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        5. Predict noise with UNet: eps_pred = UNet(xt, t, c) (trainable)
        6. Compute MSE loss: ||eps - eps_pred||^2
        7. Backpropagate and update UNet weights only
        """
        self.model.unet.train()
        self._optimizer.zero_grad()
        batch_size = images.shape[0]
        # 1. Encode images to latent space (VAE is frozen)
        with torch.no_grad():
            latents, _, _ = self.model.vae.encode(images)
            latents = latents * self.model.vae.scaling_factor
        # 2. Encode text prompts (CLIP is frozen)
        with torch.no_grad():
            tokens = self.model.text_encoder.tokenize(text_prompts)
            text_embeddings = self.model.text_encoder(tokens.input_ids.to(self._device))
        # 3. Sample random timesteps for each image in the batch
        timesteps = torch.randint(
            0,
            self.model.scheduler.num_timesteps,
            (batch_size,),
            device=self._device,
            dtype=torch.long,
        )
        # 4. Sample noise and add to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.model.scheduler.add_noise(
            latents, timesteps, noise, self._device
        )
        # 5. Predict noise with trainable UNet
        noise_pred = self.model.unet(noisy_latents, timesteps, text_embeddings)
        # 6. Compute loss (MSE between true noise and predicted noise)
        loss = F.mse_loss(noise_pred, noise)
        # 7. Backpropagation with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.unet.parameters(), self._max_grad_norm
        )
        self._optimizer.step()
        return loss.item()
