import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.utils.tools import unflatten_dict

CHECKPOINT_NAME = "CellViT-256-x40-AMP.pth"
CHECKPOINT_URL = "https://zenodo.org/records/15094831/files/CellViT-256-x40-AMP.pth"
CHECKPOINT_MD5 = "c0a794a5777320ff3a5a0e3be2e74a7c"
CHECKPOINT_SHA256 = "c58d356c92d3fd217586e387d294ddf2cc95de1ee4ffbd5cdbdcbd4bd6970238"


@dataclass(frozen=True)
class SegmentationModel:
    network: CellViT256
    nuclei_types: dict[int, str]
    tissue_types: dict[int, str]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    token_size: int


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> None:
    md5 = file_digest(path, "md5")
    sha256 = file_digest(path, "sha256")
    if md5 != CHECKPOINT_MD5 or sha256 != CHECKPOINT_SHA256:
        raise ValueError(f"Checksum mismatch for {path}: md5={md5} sha256={sha256}")


def download_checkpoint(directory: Path) -> Path:
    target = Path(directory) / CHECKPOINT_NAME
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(CHECKPOINT_URL) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        partial.replace(target)
    verify_checkpoint(target)
    return target


def load_model(checkpoint_path: Path) -> SegmentationModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["arch"] != "CellViT256":
        raise ValueError(f"Expected a CellViT256 checkpoint, got {checkpoint['arch']}")
    config = unflatten_dict(checkpoint["config"], ".")
    network = CellViT256(
        model256_path=None,
        num_nuclei_classes=config["data"]["num_nuclei_classes"],
        num_tissue_classes=config["data"]["num_tissue_classes"],
        regression_loss=config["model"].get("regression_loss", False),
    )
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.eval()
    normalize = config["transformations"].get("normalize", {})
    return SegmentationModel(
        network=network,
        nuclei_types={v: k for k, v in config["dataset_config"]["nuclei_types"].items()},
        tissue_types={v: k for k, v in config["dataset_config"]["tissue_types"].items()},
        mean=tuple(normalize.get("mean", (0.5, 0.5, 0.5))),
        std=tuple(normalize.get("std", (0.5, 0.5, 0.5))),
        token_size=network.patch_size,
    )
