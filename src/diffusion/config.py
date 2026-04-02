import torch

from .base_dataset import BaseDataset
from .datasets.caption_dataset import CaptionDataset


class Config:

    @staticmethod
    def ts_dataset(
        inputs: torch.Tensor,
        transform=None,
        use_captioner=True,
        caption_batch_size=64,
        caption_cache_file: str | None = None,
    ) -> BaseDataset:
        return CaptionDataset(
            inputs,
            transform,
            use_captioner=use_captioner,
            caption_batch_size=caption_batch_size,
            caption_cache_file=caption_cache_file,
        )
