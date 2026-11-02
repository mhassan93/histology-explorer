import argparse
from pathlib import Path

from histology_explorer.segmentation import download_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the CellViT-256-x40 checkpoint.")
    parser.add_argument("--dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    path = download_checkpoint(args.dir)
    print(f"Checkpoint ready and verified: {path}")


if __name__ == "__main__":
    main()
