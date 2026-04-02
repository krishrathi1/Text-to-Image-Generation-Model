import os
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Reasonable Kaggle defaults; CLI args can override these.
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

    # Use defaults only when user did not pass explicit args.
    train_args = sys.argv[1:] if len(sys.argv) > 1 else default_args
    sys.argv = ["unet-train"] + train_args

    from diffusion.__main__ import Main

    print("Kaggle training launch args:", " ".join(train_args))
    print("Detected CUDA devices:", os.environ.get("CUDA_VISIBLE_DEVICES", "all"))
    Main.training()


if __name__ == "__main__":
    main()
