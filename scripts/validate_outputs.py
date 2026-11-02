import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from histology_explorer.segmentation import (
    load_instance_mask,
    load_model,
    load_rgb,
    predict,
    preprocess,
    segment,
    verify_checkpoint,
)
from histology_explorer.segmentation.model import CHECKPOINT_NAME

EXPECTED_STATE_ELEMENTS = 46_764_240
EXPECTED_NUCLEI_TYPES = {0: "Background", 1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"}


class Checks:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}{': ' + detail if detail else ''}")

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.results)


def all_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(all_finite(v) for v in value)
    return True


def check_checkpoint(checks: Checks, checkpoint: Path, image: np.ndarray) -> None:
    verify_checkpoint(checkpoint)
    checks.add("checkpoint checksum (md5 + sha256)", True)
    model = load_model(checkpoint)
    checks.add("state_dict loaded with strict=True", True)
    state_elements = sum(t.numel() for t in model.network.state_dict().values())
    parameters = sum(p.numel() for p in model.network.parameters())
    checks.add(
        "state_dict element count matches checkpoint",
        state_elements == EXPECTED_STATE_ELEMENTS,
        f"{state_elements:,} elements ({parameters:,} parameters + BatchNorm buffers)",
    )
    checks.add("nuclei class mapping", model.nuclei_types == EXPECTED_NUCLEI_TYPES, str(model.nuclei_types))
    checks.add("normalisation", model.mean == (0.5, 0.5, 0.5) and model.std == (0.5, 0.5, 0.5), f"mean={model.mean} std={model.std}")
    checks.add("model in eval mode", not model.network.training)
    checks.add("CUDA not used", not torch.cuda.is_available() and all(p.device.type == "cpu" for p in model.network.parameters()))

    patch = preprocess(image, model)
    predictions = predict(model, patch)
    finite = all(bool(torch.isfinite(t).all()) for t in predictions.values())
    checks.add("network outputs finite", finite)
    shapes = {k: tuple(v.shape) for k, v in predictions.items()}
    padded_height, padded_width = patch.tensor.shape[-2:]
    expected = {
        "nuclei_binary_map": (1, padded_height, padded_width, 2),
        "nuclei_type_map": (1, padded_height, padded_width, 6),
        "hv_map": (1, padded_height, padded_width, 2),
        "tissue_types": (1, 19),
    }
    checks.add(
        "padding to a multiple of the 16 px token size",
        padded_height % model.token_size == 0
        and padded_width % model.token_size == 0
        and 0 <= padded_height - image.shape[0] < model.token_size
        and 0 <= padded_width - image.shape[1] < model.token_size,
        f"input {image.shape[0]}x{image.shape[1]} -> model {padded_height}x{padded_width}",
    )
    checks.add("network output shapes", shapes == expected, str(shapes))
    tissue = predictions["tissue_types"][0]
    index = int(torch.argmax(tissue))
    checks.add(
        "tissue head sanity (sample is breast tissue)",
        model.tissue_types[index] == "Breast",
        f"predicted {model.tissue_types[index]} p={float(tissue[index]):.3f}",
    )

    first = segment(model, image)
    second = segment(model, image)
    deterministic = np.array_equal(first.instance_mask, second.instance_mask) and first.nuclei == second.nuclei
    checks.add("repeat run gives identical mask and metadata", deterministic)


def check_outputs(checks: Checks, output_dir: Path, source_image: np.ndarray) -> None:
    image = load_rgb(output_dir / "input.png")
    overlay = load_rgb(output_dir / "overlay.png")
    mask = load_instance_mask(output_dir / "instance_mask.png")
    record = json.loads((output_dir / "nuclei.json").read_text(encoding="utf-8"))
    nuclei = record["nuclei"]

    checks.add("saved input identical to source pixels", np.array_equal(image, source_image))
    checks.add("mask and overlay match input size", mask.shape == image.shape[:2] and overlay.shape == image.shape, f"mask={mask.shape} overlay={overlay.shape}")
    ids = np.unique(mask)
    ids = ids[ids > 0]
    checks.add("mask values non-negative integers", mask.min() >= 0)
    checks.add("mask IDs contiguous 1..N", np.array_equal(ids, np.arange(1, len(ids) + 1)), f"N={len(ids)}")
    checks.add("JSON IDs equal mask IDs", sorted(n["id"] for n in nuclei) == ids.tolist())
    checks.add("nucleus_count consistent", record["nucleus_count"] == len(nuclei) == len(ids))
    checks.add("no NaN/inf in JSON", all_finite(record))

    height, width = mask.shape
    problems = []
    for n in nuclei:
        pixels = mask == n["id"]
        rows, cols = np.nonzero(pixels)
        if n["area_px"] != len(rows):
            problems.append(f"{n['id']}: area")
        if not (abs(n["centroid"][0] - cols.mean()) < 1e-6 and abs(n["centroid"][1] - rows.mean()) < 1e-6):
            problems.append(f"{n['id']}: centroid")
        if n["bbox"] != [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]:
            problems.append(f"{n['id']}: bbox")
        contour = np.asarray(n["contour"])
        if len(contour) < 3:
            problems.append(f"{n['id']}: contour<3")
        inside = (contour[:, 0] >= 0) & (contour[:, 0] < width) & (contour[:, 1] >= 0) & (contour[:, 1] < height)
        if not inside.all() or not (mask[contour[:, 1], contour[:, 0]] == n["id"]).all():
            problems.append(f"{n['id']}: contour off instance")
        if record["type_map"].get(str(n["type_id"])) != n["type_name"] or n["type_id"] == 0:
            problems.append(f"{n['id']}: type")
        if not 0 < n["type_fraction"] <= 1:
            problems.append(f"{n['id']}: type_fraction")
    checks.add("per-nucleus area, centroid, bbox, contour and type match the mask", not problems, "; ".join(problems[:10]))

    background = mask == 0
    checks.add("overlay leaves non-nucleus pixels untouched", np.array_equal(overlay[background], image[background]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the checkpoint and the saved segmentation outputs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / CHECKPOINT_NAME)
    args = parser.parse_args()

    checks = Checks()
    source_image = load_rgb(args.input)
    check_checkpoint(checks, args.checkpoint, source_image)
    check_outputs(checks, args.output_dir, source_image)

    report = {"all_passed": checks.all_passed, "checks": checks.results}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{sum(r['passed'] for r in checks.results)}/{len(checks.results)} checks passed")
    raise SystemExit(0 if checks.all_passed else 1)


if __name__ == "__main__":
    main()
