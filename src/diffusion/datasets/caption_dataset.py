import torch

from transformers import pipeline
from transformers.pipelines.image_to_text import ImageToTextPipeline
from PIL import Image
from ..base_dataset import BaseDataset


class CaptionDataset(BaseDataset):

    _captions: list[str]

    def __init__(
        self, inputs: torch.Tensor, transform=None, use_captioner=True
    ) -> None:
        super().__init__(inputs, transform)
        if use_captioner:
            print("Initializing captioner...")
            captioner = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=0 if torch.cuda.is_available() else -1,
            )
            print(f"Generating captions for {len(inputs)} images...")
            self._captions = self._generate_all_captions(captioner)
            print(f"✓ Caption generation complete!")
            del captioner
            torch.cuda.empty_cache()
        else:
            print("Using dummy captions")
            self._captions = ["a photo"] * len(inputs)

    def _generate_all_captions(self, captioner: ImageToTextPipeline) -> list[str]:
        all_captions = []
        batch_size = 32
        for i in range(0, len(self._inputs), batch_size):
            batch = self._inputs[i : i + batch_size]
            pil_images = []
            for img in batch:
                img_pil = Image.fromarray(
                    (img.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                )
                pil_images.append(img_pil)
            results = captioner(pil_images)
            captions = [r[0]["generated_text"] for r in results]
            all_captions.extend(captions)
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= len(self._inputs):
                print(
                    f"  Generated {min(i + batch_size, len(self._inputs))}/{len(self._inputs)} captions"
                )
        return all_captions

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image = super().__getitem__(index)
        caption = self._captions[index]
        return image, caption
