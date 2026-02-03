import torch

from .base_dataset import BaseDataset
from .datasets.caption_dataset import CaptionDataset


class Config:

    @staticmethod
    def ts_dataset(inputs: torch.Tensor, transform=None) -> BaseDataset:
        return CaptionDataset(inputs, transform)
