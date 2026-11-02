import time

START = time.perf_counter()

import argparse
import json
import os
import platform
from pathlib import Path

import psutil
import torch

from histology_explorer.segmentation import load_model, load_rgb, postprocess, predict, preprocess, verify_checkpoint
from histology_explorer.segmentation.model import CHECKPOINT_NAME

IMPORTS_DONE = time.perf_counter()


def timed(function, *args):
    begin = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter() - begin


def peak_rss_bytes(memory) -> int:
    if hasattr(memory, "peak_wset"):
        return memory.peak_wset
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def run_once(model, image) -> dict:
    patch, t_pre = timed(preprocess, image, model)
    predictions, t_forward = timed(predict, model, patch)
    result, t_post = timed(postprocess, model, predictions, patch)
    return {
        "preprocess_s": t_pre,
        "forward_s": t_forward,
        "postprocess_s": t_post,
        "total_s": t_pre + t_forward + t_post,
        "nuclei": len(result.nuclei),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CPU load time, inference time and peak RAM.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / CHECKPOINT_NAME)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    process = psutil.Process()

    _, t_verify = timed(verify_checkpoint, args.checkpoint)
    model, t_load = timed(load_model, args.checkpoint)
    rss_after_load = process.memory_info().rss
    image = load_rgb(args.input)

    cold = run_once(model, image)
    warm = [run_once(model, image) for _ in range(args.warm_runs)]
    memory = process.memory_info()

    report = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "settings": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "input": args.input.name,
            "input_size_hw": list(image.shape[:2]),
            "model_input_size_hw": list(preprocess(image, model).tensor.shape[-2:]),
            "precision": "float32",
            "batch_size": 1,
        },
        "startup": {
            "imports_s": IMPORTS_DONE - START,
            "checkpoint_verification_s": t_verify,
            "model_load_s": t_load,
            "first_inference_s": cold,
        },
        "warm_runs": warm,
        "memory": {
            "rss_after_model_load_mb": round(rss_after_load / 2**20, 1),
            "rss_at_end_mb": round(memory.rss / 2**20, 1),
            "peak_rss_mb": round(peak_rss_bytes(memory) / 2**20, 1),
            "peak_metric": "Windows peak working set" if hasattr(memory, "peak_wset") else "getrusage ru_maxrss",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"threads={report['settings']['torch_num_threads']} input={image.shape[1]}x{image.shape[0]}")
    print(f"imports {report['startup']['imports_s']:.2f}s | verify {t_verify:.2f}s | model load {t_load:.2f}s | first inference {cold['total_s']:.2f}s")
    for i, run in enumerate(warm, start=1):
        print(f"warm run {i}: total {run['total_s']:.3f}s (pre {run['preprocess_s']:.3f} / forward {run['forward_s']:.3f} / post {run['postprocess_s']:.3f})")
    print(f"peak process RAM {report['memory']['peak_rss_mb']} MB ({report['memory']['peak_metric']})")


if __name__ == "__main__":
    main()
