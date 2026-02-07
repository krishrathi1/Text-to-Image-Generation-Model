import pytest
import torch

from scipy import stats
from diffusion.vae import VAE


# pytest -sv tests/diffusion/test_vae.py
class TestVAE:

    @pytest.fixture
    def vae(self) -> VAE:
        return VAE()

    def test_aggregated_latent_distribution_is_gaussian(self, vae: VAE) -> None:
        vae.eval()
        num_samples = 200
        all_latents = []
        for _ in range(num_samples):
            img = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                latents, _, _ = vae.encode(img)
            all_latents.append(latents)
        all_latents = torch.cat(all_latents, dim=0)  # (200, 4, 32, 32)
        latents_flat = all_latents.flatten().cpu().numpy()
        latents_mean = latents_flat.mean()
        latents_std = latents_flat.std()
        latents_standardized = (latents_flat - latents_mean) / latents_std
        """
        Anderson-Darling test
        """
        res = stats.anderson(latents_standardized, dist="norm", method="interpolate")
        pvalue = res.pvalue  # type: ignore[attr-defined]
        print(f"\nAggregated latent distribution:")
        print(f"  Mean: {latents_mean:.6f}")
        print(f"  Std: {latents_std:.6f}")
        print(f"  P-value: {pvalue:.6f}")
        assert (
            pvalue < 0.025
        ), f"Aggregated latents don't follow Gaussian: AD={pvalue:.4f} >= 0.025"
