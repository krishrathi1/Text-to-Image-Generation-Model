import torch
import torch.nn as nn

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class CLIPTextEncoder(nn.Module):

    _tokenizer: CLIPTokenizer
    _transformer: CLIPTextModel

    def __init__(self, model_name="openai/clip-vit-large-patch14") -> None:
        super().__init__()
        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._transformer = CLIPTextModel.from_pretrained(model_name)
        """
        freeze CLIP weights
        """
        for param in self._transformer.parameters():
            param.requires_grad = False

    def tokenize(self, text: str | list[str]) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]
        return self._tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self._transformer(input_ids=input_ids)
        return outputs.last_hidden_state  # Shape: (batch_size, 77, 768)
