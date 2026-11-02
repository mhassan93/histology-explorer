import argparse
import hashlib
from collections import Counter
from pathlib import Path

import torch

from histology_explorer.segmentation import load_model, load_rgb, save_outputs, segment, verify_checkpoint
from histology_explorer.segmentation.model import CHECKPOINT_NAME, CHECKPOINT_SHA256, CHECKPOINT_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment nuclei in one H&E patch with CellViT-256 on CPU.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / CHECKPOINT_NAME)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--mpp", type=float, default=None)
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    verify_checkpoint(args.checkpoint)
    model = load_model(args.checkpoint)
    image = load_rgb(args.input)
    result = segment(model, image)

    source = {
        "file": args.input.name,
        "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "height": image.shape[0],
        "width": image.shape[1],
        "mpp": args.mpp,
        "resized": False,
    }
    model_info = {
        "name": "CellViT-256-x40 (HIPT ViT-256 encoder)",
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "device": "cpu",
        "precision": "float32",
        "torch_threads": torch.get_num_threads(),
    }
    paths = save_outputs(args.output_dir, image, result, source, model_info, model.nuclei_types)

    counts = Counter(n["type_name"] for n in result.nuclei)
    print(f"Input: {args.input} ({image.shape[1]}x{image.shape[0]} px)")
    print(f"Nuclei: {len(result.nuclei)} {dict(sorted(counts.items()))}")
    print(f"Discarded instances: {result.discarded_instances}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
