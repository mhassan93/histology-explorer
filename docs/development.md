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

## Speed and memory

Measured with PyTorch configured to use four CPU threads:
- a 512 × 512 crop took about 6 s (7–9 s in Docker), and process memory peaked at 1.8 GB;
- a 256 × 256 patch peaked at about 0.8 GB.

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

Open http://127.0.0.1:7860. The model loads once at startup.
- **Choosing a crop:** on a larger image, click two opposite corners on the preview, or type x, y,
  width and height. Crops that are too small, too large or past the image edge are outlined in
  red with an explanation, and **Run segmentation** refuses them.
- **Replacing the image:** drop a new file onto the upload box, or click the box to choose one;
  there is no need to press × first. The previous crop, result, counts and nucleus details are
  cleared at once, and any segmentation still running for the old image is cancelled.
- **Unreadable files:** an unreadable or unsupported file leaves the interface empty, with the
  reason in the status line.
- **No recomputation:** toggling the overlay or clicking a nucleus reuses the stored result
  without running the model again.

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

## How the Docker setup works

| Part | What it is | What it does here |
|---|---|---|
| **Image** `histology-explorer:local` | A read-only package built from the `Dockerfile` | Holds Python, the pinned CPU-only libraries, the app code and the sample patch. It does not hold the model checkpoint. |
| **Containers** `checkpoint` and `app` | Running copies of that image, defined in `compose.yaml` | `checkpoint` downloads and verifies the model file, then exits; it runs as root so it can write into a newly created `checkpoints/` folder. `app` runs the browser interface on port 7860 as a non-root user. `docker compose down` removes both; the image stays. |
| **Mounted folder** `checkpoints/` | The project folder on your computer, shared into the containers at `/app/checkpoints` | Stores the 187 MB checkpoint outside the image, so it is downloaded once and survives rebuilds and `down`. `app` sees it read-only. |

`docker compose up --build` runs these steps in order:
1. **Build the image.** The first build downloads the Python base image and roughly 450 MB of
   packages; later builds reuse Docker's cache.
2. **Run `checkpoint`.** It downloads the checkpoint into `checkpoints/` if it is missing, verifies
   it either way, and exits with code 0.
3. **Start `app`.** This happens only if the verification succeeded.

The app is published only on this computer (`127.0.0.1`), not on your network. The log also prints
`http://0.0.0.0:7860`, which is the address inside the container, so use `localhost` instead.

A checkpoint already downloaded by `scripts\download_checkpoint.py` is reused. To force a fresh
download, delete the file and start again. On Linux the downloaded file is owned by root (as is
`checkpoints/` if Docker created it), so deleting it needs `sudo`.

`docker compose up --build -d` runs it in the background, and `docker compose logs -f app` shows
the log. Code changes need `--build` to take effect.

The published image uses the same layout, with a Docker volume named
`histology-explorer-checkpoints` instead of the `checkpoints/` folder. To delete that downloaded
checkpoint, run `docker volume rm histology-explorer-checkpoints`. A workflow in
`.github/workflows/publish-image.yml` builds the image from each `v*.*.*` tag and pushes it to
`ghcr.io`.

## Project layout

```
src/histology_explorer/                browser interface and image input checks
src/histology_explorer/segmentation/   model loading, pipeline, output writing
scripts/                               command-line entry points
third_party/cellvit_inference/         vendored upstream model code (see its README)
data/samples/                          sample patch with provenance
docs/                                  this guide
```
