import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as TF

from datasets import load_dataset
from datetime import datetime
from pathlib import Path
from PIL import Image
from .stable_diffusion import StableDiffusion
from .stable_diffusion_trainer import StableDiffusionTrainer
from .train_service import TrainService


class Main:

    @staticmethod
    def save_current_plot(filename: str, folder="outputs", close=True) -> Path:
        save_dir = Path.home() / "src" / "diffusion" / folder
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        plt.savefig(save_path)
        if close:
            plt.close()
        return save_path

    @staticmethod
    def training():
        start = datetime.now()
        start_time_formatted = start.strftime("%H:%M:%S")
        print("Start Time =", start_time_formatted)
        parser = argparse.ArgumentParser(
            description="Train diffusion model",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
        Examples:
        unet-train > log.txt
        unet-train --epochs 100 > second_log.txt
        """,
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=30,
            help="Number of training epochs",
        )
        args = parser.parse_args()
        """
        CLIP: Pre-trained (openai/clip-vit-large-patch14)
        VAE: Pre-trained (stabilityai/sd-vae-ft-mse)
        UNet: Train from scratch
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        assert device == torch.device(
            "cuda"
        ), "CUDA device not found. Operation requires a GPU."
        print(f"Using device: {device}")
        print("\n" + "=" * 50)
        print("TRAINING EXAMPLE - UNet from Scratch")
        print("=" * 50)
        print("Loading pre-trained CLIP and VAE...")
        model = StableDiffusion(
            clip_model_name="openai/clip-vit-large-patch14",
            vae_model_name="stabilityai/sd-vae-ft-mse",
        )
        model = model.to(device)
        print("✓ CLIP: Pre-trained (frozen)")
        print("✓ VAE: Pre-trained (frozen)")
        print("✓ UNet: Random initialization (trainable)")
        optimizer = torch.optim.AdamW(model.unet.parameters(), lr=1e-4)
        trainer = StableDiffusionTrainer(model, optimizer, device)
        dataset = load_dataset("mattymchen/celeba-hq", split="train")
        N = len(dataset)
        x0 = torch.empty((N, 3, 256, 256), dtype=torch.float32)
        for idx, item in enumerate(dataset):
            assert isinstance(item, dict)
            img = item["image"].convert("RGB")
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()
            x0[idx] = tensor
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx+1}/{N}")
        transform = TF.Compose(
            [
                TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1.0, 1.0]
            ]
        )
        TrainService(
            x0,
            device,
            transform,
            trainer,
            f"celeba_unet_1e_4_16_{args.epochs}.pt",
            batch_size=16,
            num_epochs=args.epochs,
        )
        end = datetime.now()
        end_time_formatted = end.strftime("%H:%M:%S")
        print("End Time =", end_time_formatted)

    @staticmethod
    def inference():
        start = datetime.now()
        start_time_formatted = start.strftime("%H:%M:%S")
        print("Start Time =", start_time_formatted)
        parser = argparse.ArgumentParser(
            description="Generate image from text prompt using diffusion model",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
        Examples:
        generate-image "A man with glasses and a beard"
        generate-image A woman with glasses
        generate-image A woman with long blonde hair
        generate-image "A person smiling"
        generate-image "A woman with dark hair"
        generate-image "A young woman"
        generate-image "A woman with long hair and glasses"
        generate-image "A man with short hair and glasses"
        generate-image "A man with glasses and a beard" --model celeba_unet_1e_4_16_100.pt
        generate-image "A woman with glasses" --model celeba_unet_1e_4_16_100.pt
        generate-image "A woman with long blonde hair" --model celeba_unet_1e_4_16_100.pt
        generate-image "A person smiling" --model celeba_unet_1e_4_16_100.pt
        generate-image "A woman with dark hair" --model celeba_unet_1e_4_16_100.pt
        generate-image "A young woman" --model celeba_unet_1e_4_16_100.pt
        generate-image "A woman with long hair and glasses" --model celeba_unet_1e_4_16_100.pt
        generate-image "A man with short hair and glasses" --model celeba_unet_1e_4_16_100.pt
        """,
        )
        parser.add_argument(
            "prompt",
            nargs="+",
            help="Text prompt for image generation",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=100,
            help="Number of inference steps",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="celeba_unet_1e_4_16_30.pt",
            help="Name of model",
        )
        args = parser.parse_args()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        assert device == torch.device(
            "cuda"
        ), "CUDA device not found. Operation requires a GPU."
        print(f"Using device: {device}")
        print("\n" + "=" * 50)
        print("INFERENCE EXAMPLE")
        print("=" * 50)
        print("Initializing Stable Diffusion...")
        model = StableDiffusion(
            clip_model_name="openai/clip-vit-large-patch14",
            vae_model_name="stabilityai/sd-vae-ft-mse",
        )
        model.eval()
        model = model.to(device)
        file_path = Path.home() / "src" / "diffusion" / "models" / args.model
        checkpoint = torch.load(file_path, map_location=device)
        model.unet.load_state_dict(checkpoint["unet_state_dict"])
        print("✓ Loaded trained UNet weights")
        prompt = " ".join(args.prompt)
        print(f"\nGenerating image for prompt: '{prompt}'")
        print("This may take a few minutes...")
        print("steps:", args.steps)
        with torch.no_grad():
            images = model.generate(
                prompt=prompt,
                num_steps=args.steps,
                guidance_scale=7.5,
                height=256,
                width=256,
            )
        print(f"\nGenerated image shape: {images.shape}")
        images = images.detach().cpu()[0, :, :, :]
        images = images.permute(1, 2, 0)
        """
        min-max scaling
        """
        images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
        plt.imshow(images)
        plt.axis("off")
        file_name = (
            prompt.replace(" ", "_") + "_" + str(args.steps) + "_" + args.model + ".png"
        )
        Main.save_current_plot(filename=file_name)
        end = datetime.now()
        end_time_formatted = end.strftime("%H:%M:%S")
        print("End Time =", end_time_formatted)


"""
unet-train > log.txt
unet-train --epochs 100 > second_log.txt

generate-image A man with glasses and a beard
generate-image A man with glasses and a beard --steps 50
generate-image "A man with glasses and a beard" --model celeba_unet_1e_4_16_100.pt
"""
if __name__ == "__main__":
    pass
