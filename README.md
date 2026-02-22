# Text-to-Image Diffusion Model — CelebA-HQ

Generate facial images from text prompts using a custom Stable Diffusion pipeline trained from scratch on the CelebA-HQ dataset.

---

## Project Overview

This project implements a text-guided image diffusion model that learns to generate facial images from text descriptions. The core idea is to train a UNet noise predictor from scratch while leveraging pre-trained CLIP (text encoding) and VAE (image compression) models — closely following the Stable Diffusion architecture and the DDPM (Denoising Diffusion Probabilistic Models) framework.

- UNet trained from scratch on 28,000 CelebA-HQ images
- Text-conditioned generation via Classifier-Free Guidance
- Pre-trained CLIP for text encoding and VAE for latent space encoding/decoding
- Automatic caption generation using BLIP captioning model

---

## Architecture

The pipeline consists of four main components:

```
Text Prompt
      │
      ▼
┌─────────────┐
│ CLIP Text   │  ← Pre-trained (frozen)
│ Encoder     │    openai/clip-vit-large-patch14
└─────┬───────┘
      │ text_embeddings
      ▼
┌─────────────┐     ┌─────────────┐
│   UNet      │◄────│  DDPM       │
│ (Trainable) │     │  Scheduler  │
└─────┬───────┘     └─────────────┘
      │ denoised latents
      ▼
┌─────────────┐
│   VAE       │  ← Pre-trained (frozen)
│  Decoder    │    stabilityai/sd-vae-ft-mse
└─────┬───────┘
      │
      ▼
Generated Image
```

### Component Breakdown

| Component | Role | Status |
|-----------|------|--------|
| **CLIP** (`clip-vit-large-patch14`) | Encodes text prompts into embeddings | Pre-trained, frozen |
| **VAE** (`sd-vae-ft-mse`) | Encodes images to / decodes from latent space | Pre-trained, frozen |
| **UNet** (`SDUNetV1`) | Predicts noise at each diffusion timestep | **Trained from scratch** |
| **DDPMScheduler** | Manages forward (noising) and reverse (denoising) process | Custom implementation |
| **BLIP Captioner** (`blip-image-captioning-base`) | Generates text captions for training images | Pre-trained, used at data prep |

### UNet Details

- **Input:** Noisy latent (4×32×32), timestep embedding, CLIP text embedding
- **Architecture:** Encoder–Decoder with cross-attention layers for text conditioning
- **Timestep Encoding:** Sinusoidal positional embedding → 2-layer MLP
- **Output:** Predicted noise (4×32×32)

### DDPM Training Objective

The model is trained to predict the noise added at each timestep:

```
L = E_{z0, ε, t} [ ||ε - ε_θ(z_t, t, c)||² ]
```

Where:
- `z0` — clean latent from VAE encoder
- `ε ~ N(0, I)` — random Gaussian noise
- `t` — randomly sampled timestep
- `c` — CLIP text embedding (condition)
- `z_t = √(ᾱ_t)·z0 + √(1-ᾱ_t)·ε` — noisy latent at timestep t

### Classifier-Free Guidance (Inference)

During sampling, unconditional and conditional noise predictions are combined:

```
ε_guided = ε_uncond + guidance_scale × (ε_text - ε_uncond)
```

A higher `guidance_scale` strengthens adherence to the text prompt (default: 7.5).

---

## Training Details

| Hyperparameter | Value |
|----------------|-------|
| Dataset | CelebA-HQ (28,000 images) |
| Resolution | 256 × 256 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 (CosineAnnealing scheduler) |
| Gradient Clipping | max_norm = 1.0 |
| Epochs | 30 → 100 |
| Diffusion Timesteps | 1000 |
| Beta Schedule | Linear (0.0001 → 0.02) |

### Training Pipeline

1. **Data Loading** — CelebA-HQ images are resized to 256×256 and normalized to [-1, 1]
2. **Caption Generation** — BLIP (`blip-image-captioning-base`) generates a caption per image
3. **Latent Encoding** — VAE encoder compresses each image to a 4×32×32 latent
4. **Noise Prediction Training** — UNet learns to denoise latents conditioned on CLIP text embeddings
5. **Model Checkpoint** — UNet state dict is saved after training

### Loss (100 Epochs)

```
Epoch   Avg Loss    LR
  1     0.0665      0.000100
  5     0.0466      0.000099
 10     0.0448      0.000098
 20     0.0430      0.000090
 30     0.0420      0.000079
 50     0.0409      0.000050
100     0.0392      0.000000
```

- Loss decreased **stably** from 0.0665 → 0.0392 with no spikes
- CosineAnnealing LR scheduler ensured smooth convergence
- Gradient clipping (max_norm=1.0) kept training stable throughout

---

## Generated Image Samples

All images below were generated with `num_steps=30` and `num_steps=100`.

| Prompt | 30 Epoch | 100 Epoch |
|--------|----------|-----------|
| A man with glasses and a beard | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_man_with_glasses_and_a_beard_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_man_with_glasses_and_a_beard_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |
| A man with short hair and glasses | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_man_with_short_hair_and_glasses_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_man_with_short_hair_and_glasses_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |
| A person smiling | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_person_smiling_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_person_smiling_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |
| A woman with dark hair | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_dark_hair_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_dark_hair_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |
| A woman with glasses | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_glasses_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_glasses_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |
| A woman with long blonde hair | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_long_blonde_hair_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_long_blonde_hair_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |
| A woman with long hair and glasses | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_long_hair_and_glasses_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_woman_with_long_hair_and_glasses_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |
| A young woman | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_young_woman_100_celeba_unet_1e_4_16_30.pt.png" width="200"> | <img src="https://raw.githubusercontent.com/jslblar080/reckit/refs/heads/main/src/diffusion/outputs/A_young_woman_100_celeba_unet_1e_4_16_100.pt.png" width="200"> |

### Key Observations

- **High-contrast features** (glasses, beard) are learned first and most clearly
- **Text conditioning works** — the same architecture with `guidance_scale=0.0` (unconditional) produces significantly blurrier results
- **Epoch count matters significantly** — features like "smiling" became noticeably more defined after 100 epochs; at 30 epochs, the expression was only faintly visible
- **Feature combinations generalize** — prompts combining multiple learned attributes (e.g., hair + glasses) produce coherent outputs

---

## Usage

### Prerequisites

- Python 3.12+
- PyTorch with CUDA support
- GPU with sufficient VRAM

### Installation

```bash
git clone https://github.com/jslblar080/reckit.git
cd reckit
uv sync # pip install uv
```

### Training

```bash
unet-train --epochs 100
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |

### Inference

```bash
# Basic generation
generate-image "A man with glasses and a beard"

# Custom steps
generate-image "A man with glasses and a beard" --steps 50

# Custom model checkpoint
generate-image "A man with glasses and a beard" --model celeba_unet_1e_4_16_100.pt
```

| Argument | Default | Description |
|----------|---------|-------------|
| `prompt` | (required) | Text description of the target image |
| `--steps` | 100 | Number of denoising steps |
| `--model` | celeba_unet_1e_4_16_30.pt | UNet state dict |

---

## References

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752)
- [Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)](https://arxiv.org/abs/2207.12598)
- [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Li et al., 2022)](https://arxiv.org/abs/2201.12086)
- [Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)