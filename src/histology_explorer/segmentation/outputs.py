import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .pipeline import SegmentationResult

TYPE_COLORS = {
    1: (255, 0, 0),
    2: (34, 221, 77),
    3: (35, 92, 236),
    4: (254, 255, 0),
    5: (255, 159, 68),
}
FILL_ALPHA = 0.35


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def save_rgb(image: np.ndarray, path: Path) -> None:
    Image.fromarray(np.ascontiguousarray(image, dtype=np.uint8)).save(path)


def save_instance_mask(mask: np.ndarray, path: Path) -> None:
    if mask.min() < 0 or mask.max() > np.iinfo(np.uint16).max:
        raise ValueError("Instance IDs must fit in an unsigned 16-bit PNG")
    Image.fromarray(mask.astype(np.uint16)).save(path)


def load_instance_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.int32).copy()


def render_overlay(image: np.ndarray, result: SegmentationResult) -> np.ndarray:
    fill = image.copy()
    for nucleus in result.nuclei:
        fill[result.instance_mask == nucleus["id"]] = TYPE_COLORS[nucleus["type_id"]]
    overlay = cv2.addWeighted(fill, FILL_ALPHA, image, 1 - FILL_ALPHA, 0)
    for nucleus in result.nuclei:
        contour = np.asarray(nucleus["contour"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.drawContours(overlay, [contour], -1, TYPE_COLORS[nucleus["type_id"]], 1)
    return overlay


def build_record(result: SegmentationResult, source: dict, model_info: dict, type_map: dict[int, str]) -> dict:
    return {
        "source": source,
        "model": model_info,
        "coordinate_system": "pixels; x = column, y = row, origin at the top-left of the input patch",
        "type_map": {str(k): v for k, v in sorted(type_map.items())},
        "nucleus_count": len(result.nuclei),
        "discarded_instances": result.discarded_instances,
        "nuclei": result.nuclei,
    }


def save_outputs(
    output_dir: Path,
    image: np.ndarray,
    result: SegmentationResult,
    source: dict,
    model_info: dict,
    type_map: dict[int, str],
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "input": output_dir / "input.png",
        "instance_mask": output_dir / "instance_mask.png",
        "overlay": output_dir / "overlay.png",
        "nuclei": output_dir / "nuclei.json",
    }
    save_rgb(image, paths["input"])
    save_instance_mask(result.instance_mask, paths["instance_mask"])
    save_rgb(render_overlay(image, result), paths["overlay"])
    record = build_record(result, source, model_info, type_map)
    paths["nuclei"].write_text(json.dumps(record, indent=2), encoding="utf-8")
    return paths
