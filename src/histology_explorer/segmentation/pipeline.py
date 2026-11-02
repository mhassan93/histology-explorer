from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from cellvit.inference.postprocessing_numpy import DetectionCellPostProcessor

from .model import SegmentationModel

PAD_VALUE = 255
BACKGROUND_TYPE = 0


@dataclass
class PreparedPatch:
    tensor: torch.Tensor
    height: int
    width: int


@dataclass
class SegmentationResult:
    instance_mask: np.ndarray
    nuclei: list[dict]
    tissue_type: str
    tissue_probability: float
    discarded_instances: dict[str, int]


def preprocess(image: np.ndarray, model: SegmentationModel) -> PreparedPatch:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB uint8 image of shape (H, W, 3), got {image.dtype} {image.shape}")
    height, width = image.shape[:2]
    padded_height = -(-height // model.token_size) * model.token_size
    padded_width = -(-width // model.token_size) * model.token_size
    padded = np.pad(
        image,
        ((0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="constant",
        constant_values=PAD_VALUE,
    )
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=model.mean, std=model.std)])
    tensor = transform(padded).unsqueeze(0)
    return PreparedPatch(tensor=tensor, height=height, width=width)


def predict(model: SegmentationModel, patch: PreparedPatch) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model.network(patch.tensor)
    return {
        "nuclei_binary_map": F.softmax(output["nuclei_binary_map"], dim=1).permute(0, 2, 3, 1),
        "nuclei_type_map": F.softmax(output["nuclei_type_map"], dim=1).permute(0, 2, 3, 1),
        "hv_map": output["hv_map"].permute(0, 2, 3, 1),
        "tissue_types": F.softmax(output["tissue_types"], dim=1),
    }


def postprocess(
    model: SegmentationModel, predictions: dict[str, torch.Tensor], patch: PreparedPatch
) -> SegmentationResult:
    processor = DetectionCellPostProcessor(wsi=None, nr_types=len(model.nuclei_types))
    instance_maps, cell_dicts = processor.post_process_batch(predictions)
    raw_instances = instance_maps[0].numpy().astype(np.int32)[: patch.height, : patch.width]
    cells = cell_dicts[0]

    raw_ids = set(np.unique(raw_instances).tolist()) - {0}
    discarded = {
        "no_valid_contour": len(raw_ids - set(int(i) for i in cells)),
        "background_type": 0,
        "centroid_in_padding": 0,
    }

    instance_mask = np.zeros((patch.height, patch.width), dtype=np.int32)
    nuclei = []
    for raw_id in sorted(int(i) for i in cells):
        cell = cells[raw_id]
        x, y = float(cell["centroid"][0]), float(cell["centroid"][1])
        if cell["type"] == BACKGROUND_TYPE:
            discarded["background_type"] += 1
            continue
        if not (0 <= x < patch.width and 0 <= y < patch.height):
            discarded["centroid_in_padding"] += 1
            continue
        nucleus_id = len(nuclei) + 1
        pixels = raw_instances == raw_id
        instance_mask[pixels] = nucleus_id
        contour = np.asarray(cell["contour"], dtype=np.int64)
        contour[:, 0] = np.clip(contour[:, 0], 0, patch.width - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, patch.height - 1)
        (row_min, col_min), (row_max, col_max) = np.asarray(cell["bbox"]).tolist()
        nuclei.append(
            {
                "id": nucleus_id,
                "centroid": [x, y],
                "bbox": [int(col_min), int(row_min), int(min(col_max, patch.width)), int(min(row_max, patch.height))],
                "area_px": int(pixels.sum()),
                "contour": contour.tolist(),
                "type_id": int(cell["type"]),
                "type_name": model.nuclei_types[int(cell["type"])],
                "type_fraction": float(cell["type_prob"]),
            }
        )

    tissue_probs = predictions["tissue_types"][0]
    tissue_index = int(torch.argmax(tissue_probs))
    return SegmentationResult(
        instance_mask=instance_mask,
        nuclei=nuclei,
        tissue_type=model.tissue_types[tissue_index],
        tissue_probability=float(tissue_probs[tissue_index]),
        discarded_instances=discarded,
    )


def segment(model: SegmentationModel, image: np.ndarray) -> SegmentationResult:
    patch = preprocess(image, model)
    predictions = predict(model, patch)
    return postprocess(model, predictions, patch)
