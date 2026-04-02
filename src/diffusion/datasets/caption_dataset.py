import torch

from pathlib import Path
from transformers import pipeline
from transformers.pipelines.image_to_text import ImageToTextPipeline
from PIL import Image
from ..base_dataset import BaseDataset


class CaptionDataset(BaseDataset):

    _captions: list[str]

    def __init__(
        self,
        inputs: torch.Tensor,
        transform=None,
        use_captioner=True,
        caption_batch_size=64,
        caption_cache_file: str | None = None,
    ) -> None:
        super().__init__(inputs, transform)
        self._caption_batch_size = caption_batch_size
        cache_path = Path(caption_cache_file) if caption_cache_file else None

        if use_captioner:
            if cache_path and cache_path.exists():
                print(f"Loading cached captions from {cache_path}...")
                captions = torch.load(cache_path)
                if len(captions) == len(inputs):
                    self._captions = captions
                    print("Loaded cached captions")
                    return
                print("Cached caption count mismatch, regenerating captions...")

            print("Initializing captioner...")
            captioner = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=0 if torch.cuda.is_available() else -1,
            )
            print(f"Generating captions for {len(inputs)} images...")
            self._captions = self._generate_all_captions(captioner)

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._captions, cache_path)
                print(f"Saved captions cache to {cache_path}")

            print("Caption generation complete")
            del captioner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("Using dummy captions")
            self._captions = ["a photo"] * len(inputs)

    def _generate_all_captions(self, captioner: ImageToTextPipeline) -> list[str]:
        all_captions = []
        batch_size = self._caption_batch_size
        for i in range(0, len(self._inputs), batch_size):
            batch = self._inputs[i : i + batch_size]
            pil_images = []
            for img in batch:
                # x0 is already in [0, 255], so do not scale by 255 again.
                img_pil = Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype("uint8"))
                pil_images.append(img_pil)

            with torch.no_grad():
                results = captioner(pil_images, batch_size=batch_size)

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
