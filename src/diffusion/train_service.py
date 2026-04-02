import torch
import torchvision.transforms as T

from pathlib import Path
from torch.utils.data import DataLoader
from .config import Config
from .stable_diffusion_trainer import StableDiffusionTrainer


class TrainService:

    def __init__(
        self,
        x0: torch.Tensor,
        device: torch.device,
        transform: T.Compose,
        trainer: StableDiffusionTrainer,
        filename: str,
        batch_size=16,
        num_epochs=30,
        folder="models",
        use_captioner=True,
        caption_batch_size=64,
        caption_cache_file: str | None = None,
        num_workers=0,
        pin_memory=False,
    ) -> None:
        print(f"Batch Size: {batch_size}")
        print(f"Number of epochs: {num_epochs}")
        dataset = Config.ts_dataset(
            x0,
            transform,
            use_captioner=use_captioner,
            caption_batch_size=caption_batch_size,
            caption_cache_file=caption_cache_file,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer._optimizer, T_max=num_epochs * len(dataloader)
        )
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            for images, captions in dataloader:
                images = images.to(device)
                loss = trainer.train_step(images, captions)
                total_loss += loss
                num_batches += 1
                scheduler.step()
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
            )
        save_dir = Path.cwd() / "src" / "diffusion" / folder
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename
        unet = trainer.model.unet
        if isinstance(unet, torch.nn.DataParallel):
            state_dict = unet.module.state_dict()
        else:
            state_dict = unet.state_dict()
        torch.save({"unet_state_dict": state_dict}, save_path)
