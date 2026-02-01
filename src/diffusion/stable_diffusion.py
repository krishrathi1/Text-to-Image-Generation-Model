import torch
import torch.nn as nn

from .vae import VAE
from .clip_text_encoder import CLIPTextEncoder
from .ddpm_scheduler import DDPMScheduler
from .sd_unet_v1 import SDUNetV1


class StableDiffusion(nn.Module):

    text_encoder: CLIPTextEncoder
    scheduler: DDPMScheduler
    unet: SDUNetV1
    vae: VAE

    def __init__(
        self,
        clip_model_name="openai/clip-vit-large-patch14",
        vae_model_name="stabilityai/sd-vae-ft-mse",
    ) -> None:
        super().__init__()
        self.text_encoder = CLIPTextEncoder(clip_model_name)
        self.scheduler = DDPMScheduler()
        self.unet = SDUNetV1()
        self.vae = VAE(vae_model_name)

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        num_steps=100,
        guidance_scale=7.5,
        height=256,
        width=256,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        tokens = self.text_encoder.tokenize(prompt)
        text_embeddings = self.text_encoder(tokens.input_ids.to(device))
        uncond_tokens = self.text_encoder.tokenize([""] * batch_size)
        uncond_embeddings = self.text_encoder(uncond_tokens.input_ids.to(device))
        latents = torch.randn(batch_size, 4, height // 8, width // 8, device=device)
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1,  # 999
            0,
            num_steps,
            dtype=torch.long,
            device=device,
        )
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t.item(), device=device)
            latent_input = torch.cat([latents] * 2)
            t_input = torch.cat([t_batch] * 2)
            context = torch.cat([uncond_embeddings, text_embeddings])
            noise_pred = self.unet(latent_input, t_input, context)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.scheduler.step(noise_pred, t_batch[0], latents)
            if (i + 1) % 10 == 0:
                print(f"Step {i+1}/{num_steps}")
        images = self.vae.decode(latents)
        images = (images / 2 + 0.5).clamp(0, 1)
        return images
