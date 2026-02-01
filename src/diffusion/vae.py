import torch
import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput


class VAE(nn.Module):

    _vae: AutoencoderKL
    scaling_factor: float

    def __init__(self, model_name="stabilityai/sd-vae-ft-mse") -> None:
        super().__init__()
        self._vae = AutoencoderKL.from_pretrained(model_name)
        for param in self._vae.parameters():
            param.requires_grad = False
        self.scaling_factor = 0.18215

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images to latent space"""
        autoencoder_kl_output = self._vae.encode(x)
        assert isinstance(autoencoder_kl_output, AutoencoderKLOutput)
        latent_dist = autoencoder_kl_output.latent_dist
        latents = latent_dist.sample()
        mean = latent_dist.mean
        logvar = 2 * torch.log(latent_dist.std)
        return latents, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to images"""
        z_scaled = (z / self.scaling_factor).float()
        decoder_output = self._vae.decode(z_scaled)  # type: ignore
        assert isinstance(decoder_output, DecoderOutput)
        decoded = decoder_output.sample
        return decoded
