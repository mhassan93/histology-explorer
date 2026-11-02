import argparse
import time
from pathlib import Path

import torch

from histology_explorer.interface import CSS, JS, MAX_UPLOAD_SIZE, build_interface
from histology_explorer.segmentation import load_model, verify_checkpoint
from histology_explorer.segmentation.model import CHECKPOINT_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local nuclei segmentation interface.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints") / CHECKPOINT_NAME)
    parser.add_argument("--sample", type=Path, default=Path("data") / "samples" / "nucls_test_2.png")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    start = time.perf_counter()
    verify_checkpoint(args.checkpoint)
    model = load_model(args.checkpoint)
    print(f"[startup] model loaded once in {time.perf_counter() - start:.2f} s ({torch.get_num_threads()} CPU threads)", flush=True)

    sample = args.sample.resolve()
    demo = build_interface(model, sample)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        css=CSS,
        js=JS,
        allowed_paths=[str(sample.parent)],
        max_file_size=MAX_UPLOAD_SIZE,
    )


if __name__ == "__main__":
    main()
