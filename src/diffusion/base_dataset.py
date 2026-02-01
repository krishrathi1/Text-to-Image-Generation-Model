import torch
import torchvision.transforms as T

from torch.utils.data import Dataset


class BaseDataset(Dataset):

    _inputs: torch.Tensor
    _transform: T.Compose | None

    def __init__(self, inputs: torch.Tensor, transform=None) -> None:
        self._inputs = inputs
        self._transform = transform

    def __getitem__(self, index: int) -> torch.Tensor:
        return (
            self._transform(self._inputs[index])
            if self._transform
            else self._inputs[index]
        )

    def __len__(self) -> int:
        return self._inputs.shape[0]
