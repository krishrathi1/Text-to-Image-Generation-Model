import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
        save_dir = Path.cwd() / "src" / "diffusion" / folder
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
        parser.add_argument(
            "--max-images",
            type=int,
            default=0,
            help="Limit number of training images (0 uses full dataset)",
        )
        parser.add_argument(
            "--use-dummy-captions",
            action="store_true",
            help="Skip BLIP caption generation and use 'a photo' for all images (faster, lower quality)",
        )
        parser.add_argument(
            "--caption-batch-size",
            type=int,
            default=64,
            help="Batch size for BLIP caption generation",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            help="Training batch size",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="DataLoader workers",
        )
        parser.add_argument(
            "--single-gpu",
            action="store_true",
            help="Disable multi-GPU DataParallel even when multiple CUDA devices are available",
        )
        args = parser.parse_args()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert device == torch.device("cuda"), "CUDA device not found. Operation requires a GPU."
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

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and not args.single_gpu:
            print(f"Using {gpu_count} GPUs with DataParallel for UNet")
            model.unet = nn.DataParallel(model.unet)

        print("[OK] CLIP: Pre-trained (frozen)")
        print("[OK] VAE: Pre-trained (frozen)")
        print("[OK] UNet: Random initialization (trainable)")

        optimizer = torch.optim.AdamW(model.unet.parameters(), lr=1e-4)
        trainer = StableDiffusionTrainer(model, optimizer, device)

        dataset = load_dataset("mattymchen/celeba-hq", split="train")
        N = len(dataset) if args.max_images <= 0 else min(len(dataset), args.max_images)
        if args.max_images > 0:
            print(f"Using subset: {N}/{len(dataset)} images")

        x0 = torch.empty((N, 3, 256, 256), dtype=torch.float32)
        for idx, item in enumerate(dataset):
            if idx >= N:
                break
            assert isinstance(item, dict)
            img = item["image"].convert("RGB")
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float()
            x0[idx] = tensor
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx+1}/{N}")

        transform = TF.Compose(
            [
                TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        TrainService(
            x0,
            device,
            transform,
            trainer,
            f"celeba_unet_1e_4_16_{args.epochs}.pt",
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            use_captioner=not args.use_dummy_captions,
            caption_batch_size=args.caption_batch_size,
            caption_cache_file=str(
                Path.cwd() / "src" / "diffusion" / "cache" / f"captions_celeba_{N}.pt"
            ),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
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
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert device == torch.device("cuda"), "CUDA device not found. Operation requires a GPU."
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
        file_path = Path.cwd() / "src" / "diffusion" / "models" / args.model
        checkpoint = torch.load(file_path, map_location=device)
        model.unet.load_state_dict(checkpoint["unet_state_dict"])
        print("[OK] Loaded trained UNet weights")
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


if __name__ == "__main__":
    pass
