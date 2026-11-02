# Development

## Local Python setup

Tested on Windows 10 with Python 3.13.14 in PowerShell. You need an internet connection for the
187 MB checkpoint download.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts\download_checkpoint.py
```

`download_checkpoint.py` fetches `CellViT-256-x40-AMP.pth` from the authors' Zenodo record
into `checkpoints/` and verifies its MD5 and SHA-256.

## Command-line tool

```powershell
.\.venv\Scripts\python.exe scripts\segment_patch.py --input data\samples\nucls_test_2.png --output-dir outputs\nucls_test_2 --mpp 0.25
```

| Output | Content |
|---|---|
| `input.png` | The input patch, re-saved losslessly |
| `instance_mask.png` | 16-bit PNG; 0 is background and 1..N are nucleus IDs |
| `overlay.png` | Nuclei filled and outlined by predicted class |
| `nuclei.json` | Per-nucleus ID, centroid, bbox, area, contour, class and model/source metadata |

`--threads N` sets the number of PyTorch CPU threads. `--mpp` only records the image scale in
the JSON; the patch is never resized.

## Browser interface

```powershell
.\.venv\Scripts\python.exe scripts\run_app.py
```

Open http://127.0.0.1:7860.

## Verification

```powershell
.\.venv\Scripts\python.exe scripts\validate_outputs.py --input data\samples\nucls_test_2.png --output-dir outputs\nucls_test_2
.\.venv\Scripts\python.exe scripts\compare_upstream.py --input data\samples\nucls_test_2.png --report outputs\nucls_test_2\upstream_comparison.json
.\.venv\Scripts\python.exe scripts\benchmark.py --input data\samples\nucls_test_2.png --report outputs\nucls_test_2\benchmark.json
```

- **`validate_outputs.py`** checks the checkpoint, the model loading and the outputs' alignment and
  consistency.
- **`compare_upstream.py`** needs internet access. It compares each stage against pristine upstream
  code at the pinned commit.
- **`benchmark.py`** measures model loading time, warm inference time and peak process RAM in a
  fresh process.

## Docker notes

- **The app does not start:** the `checkpoint` service must download and verify the model before
  `app` starts. Run `docker compose logs checkpoint` to see why it failed.
- **Fresh download:** an existing `checkpoints/CellViT-256-x40-AMP.pth` is reused, including one
  from the local setup. Delete it to force a new download. On Linux the file is owned by root (as
  is `checkpoints/` if Docker created it), so deleting it needs `sudo`.
- **Access from other machines:** the port is bound to `127.0.0.1`, so the app is reachable only
  from this computer.
- **Running in the background:** `docker compose up --build -d` starts it detached, and
  `docker compose logs -f app` shows the log. Code changes need `--build` to take effect.
- **Published image:** its checkpoint lives in the Docker volume `histology-explorer-checkpoints`;
  `docker volume rm histology-explorer-checkpoints` deletes it.
