import os
import sys

from .__main__ import Main


def main() -> None:
    default_args = [
        "--epochs",
        "5",
        "--max-images",
        "8000",
        "--batch-size",
        "32",
        "--num-workers",
        "2",
        "--caption-batch-size",
        "96",
    ]
    train_args = sys.argv[1:] if len(sys.argv) > 1 else default_args
    sys.argv = ["unet-train"] + train_args
    print("Kaggle training launch args:", " ".join(train_args))
    print("Detected CUDA devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))
    Main.training()
