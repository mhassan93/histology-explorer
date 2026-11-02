import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import urllib.request
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
from cellvit.utils.tools import unflatten_dict
from histology_explorer.segmentation import load_model, load_rgb, postprocess, predict, preprocess
from histology_explorer.segmentation.model import CHECKPOINT_NAME
from histology_explorer.segmentation.outputs import TYPE_COLORS

UPSTREAM_REPO = "TIO-IKIM/CellViT-Inference"
UPSTREAM_COMMIT = "d2143ded642846c6265fcfda9711a0376a894d8e"
VENDOR_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "cellvit_inference"
MODIFIED_FILES = {"cellvit/inference/postprocessing_numpy.py"}


def fetch(path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/{path}"
    with urllib.request.urlopen(url) as response:
        return response.read()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def upstream_tree() -> dict[str, str]:
    url = f"https://api.github.com/repos/{UPSTREAM_REPO}/git/trees/{UPSTREAM_COMMIT}?recursive=1"
    with urllib.request.urlopen(url) as response:
        tree = json.load(response)
    return {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}


def extract_method(source: str, class_name: str, method_name: str, namespace: dict):
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    module = ast.Module(body=[item], type_ignores=[])
                    exec(compile(module, f"upstream:{class_name}.{method_name}", "exec"), namespace)
                    return namespace[method_name]
    raise LookupError(f"{class_name}.{method_name} not found upstream")


def load_pristine_postprocessing(path: Path):
    ray_stub = types.ModuleType("ray")
    ray_stub.remote = lambda *args, **kwargs: (lambda cls: cls)
    wsi_stub = types.ModuleType("cellvit.data.dataclass.wsi")
    wsi_stub.WSI = wsi_stub.WSIMetadata = object
    stubs = {
        "ray": ray_stub,
        "cellvit.data": types.ModuleType("cellvit.data"),
        "cellvit.data.dataclass": types.ModuleType("cellvit.data.dataclass"),
        "cellvit.data.dataclass.wsi": wsi_stub,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("upstream_postprocessing_numpy", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def cells_equal(a: dict, b: dict) -> bool:
    if sorted(a) != sorted(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        if x["type"] != y["type"] or x["type_prob"] != y["type_prob"]:
            return False
        for field in ("bbox", "centroid", "contour"):
            if not np.array_equal(np.asarray(x[field]), np.asarray(y[field])):
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the patch pipeline with pristine upstream CellViT-Inference code.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / CHECKPOINT_NAME)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    results: list[dict] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        results.append({"comparison": name, "as_expected": bool(passed), "detail": detail})
        print(f"[{'OK' if passed else 'MISMATCH'}] {name}{': ' + detail if detail else ''}")

    tree = upstream_tree()
    vendored = ["LICENSE"] + sorted(
        p.relative_to(VENDOR_ROOT).as_posix() for p in (VENDOR_ROOT / "cellvit").rglob("*.py") if "__pycache__" not in p.parts
    )
    for relative in vendored:
        local_sha = git_blob_sha((VENDOR_ROOT / relative).read_bytes())
        expected_same = relative not in MODIFIED_FILES
        record(f"vendored file {relative} {'matches' if expected_same else 'differs from'} upstream blob", (local_sha == tree.get(relative)) == expected_same, local_sha)

    with tempfile.TemporaryDirectory() as scratch:
        pristine_post = Path(scratch) / "postprocessing_numpy.py"
        pristine_post.write_bytes(fetch("cellvit/inference/postprocessing_numpy.py"))
        inference_source = fetch("cellvit/inference/inference.py").decode("utf-8")
        config_source = fetch("cellvit/config/config.py").decode("utf-8")
        diff = "".join(
            difflib.unified_diff(
                pristine_post.read_text(encoding="utf-8").splitlines(keepends=True),
                (VENDOR_ROOT / "cellvit/inference/postprocessing_numpy.py").read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile="upstream/cellvit/inference/postprocessing_numpy.py",
                tofile="vendored/cellvit/inference/postprocessing_numpy.py",
            )
        )
        upstream_post = load_pristine_postprocessing(pristine_post)

    constants = {}
    for node in ast.parse(config_source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("TYPE_NUCLEI_DICT_PANNUKE", "COLOR_DICT_CELLS"):
            constants[node.targets[0].id] = ast.literal_eval(node.value)

    model = load_model(args.checkpoint)
    image = load_rgb(args.input)

    record(
        "nuclei class mapping vs upstream TYPE_NUCLEI_DICT_PANNUKE",
        {k: v for k, v in model.nuclei_types.items() if k != 0} == constants["TYPE_NUCLEI_DICT_PANNUKE"],
    )
    record(
        "overlay colours vs upstream COLOR_DICT_CELLS",
        all(list(TYPE_COLORS[k]) == constants["COLOR_DICT_CELLS"][k] for k in TYPE_COLORS),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    run_conf = unflatten_dict(checkpoint["config"], ".")
    upstream_self = types.SimpleNamespace(run_conf=run_conf, logger=types.SimpleNamespace(info=lambda *a, **k: None))

    load_transforms = extract_method(inference_source, "CellViTInference", "_load_inference_transforms", {"T": T})
    load_transforms(upstream_self)
    reference_tensor = upstream_self.inference_transforms(image).unsqueeze(0)
    patch = preprocess(image, model)
    record("preprocessed tensor vs upstream _load_inference_transforms", torch.equal(patch.tensor, reference_tensor))

    get_model = extract_method(
        inference_source,
        "CellViTInference",
        "_get_model",
        {"CellViT256": CellViT256, "CellViTSAM": None, "Literal": Literal, "Union": Union},
    )
    reference_network = get_model(upstream_self, model_type=checkpoint["arch"])
    reference_network.load_state_dict(checkpoint["model_state_dict"])
    reference_network.eval()
    with torch.no_grad():
        reference_raw = reference_network.forward(reference_tensor, retrieve_tokens=True)
        own_raw = model.network(patch.tensor)
    same_forward = all(torch.equal(reference_raw[k], own_raw[k]) for k in ("nuclei_binary_map", "hv_map", "nuclei_type_map", "tissue_types"))
    record("raw network outputs vs model built by upstream _get_model", same_forward)

    apply_softmax_reorder = extract_method(inference_source, "CellViTInference", "apply_softmax_reorder", {"F": F})
    reference_predictions = apply_softmax_reorder(None, {k: v.clone() for k, v in reference_raw.items()})
    own_predictions = predict(model, patch)
    record(
        "softmax/reorder vs upstream apply_softmax_reorder",
        all(torch.equal(reference_predictions[k], own_predictions[k]) for k in ("nuclei_binary_map", "nuclei_type_map", "hv_map")),
    )

    reference_processor = upstream_post.DetectionCellPostProcessor(wsi=None, nr_types=run_conf["data"]["num_nuclei_classes"])
    reference_instances, reference_cells = reference_processor.post_process_batch(reference_predictions)
    from cellvit.inference.postprocessing_numpy import DetectionCellPostProcessor

    vendored_processor = DetectionCellPostProcessor(wsi=None, nr_types=run_conf["data"]["num_nuclei_classes"])
    vendored_instances, vendored_cells = vendored_processor.post_process_batch(own_predictions)
    record("instance map: pristine vs vendored postprocessor", torch.equal(reference_instances, vendored_instances))
    record("cell dict: pristine vs vendored postprocessor", cells_equal(reference_cells[0], vendored_cells[0]))

    result = postprocess(model, own_predictions, patch)
    reference_kept = {k: v for k, v in reference_cells[0].items() if v["type"] != run_conf["dataset_config"]["nuclei_types"]["Background"]}
    reference_mask = np.zeros_like(result.instance_mask)
    raw = reference_instances[0].numpy().astype(np.int32)
    for new_id, raw_id in enumerate(sorted(reference_kept), start=1):
        reference_mask[raw == raw_id] = new_id
    record("final instance mask vs pristine upstream postprocessing", np.array_equal(result.instance_mask, reference_mask), f"{len(result.nuclei)} nuclei")
    same_records = len(result.nuclei) == len(reference_kept) and all(
        n["type_id"] == reference_kept[raw_id]["type"]
        and n["type_fraction"] == reference_kept[raw_id]["type_prob"]
        and n["centroid"] == [float(c) for c in reference_kept[raw_id]["centroid"]]
        and n["contour"] == np.asarray(reference_kept[raw_id]["contour"]).tolist()
        for n, raw_id in zip(result.nuclei, sorted(reference_kept))
    )
    record("final nucleus records vs pristine upstream postprocessing", same_records)

    report = {
        "upstream": f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT}",
        "all_as_expected": all(r["as_expected"] for r in results),
        "comparisons": results,
        "postprocessing_numpy_diff": diff,
        "not_compared": [
            "End-to-end CellViTInference.process_wsi: hard-codes a CUDA device and depends on Ray, pathopatch and OpenSlide.",
            "Upstream fp16 autocast on CUDA (checkpoint trained with mixed precision); this pipeline runs fp32 on CPU.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{sum(r['as_expected'] for r in results)}/{len(results)} comparisons as expected")
    print(diff)
    raise SystemExit(0 if report["all_as_expected"] else 1)


if __name__ == "__main__":
    main()
