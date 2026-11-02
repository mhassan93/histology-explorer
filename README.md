# Histology Explorer

Nuclei instance segmentation for H&E image patches with CellViT-256 (HIPT ViT-256 encoder),
running entirely on CPU on a vendored subset of the official CellViT-Inference code. CellViT-256's
own PanNuke-trained type head labels each nucleus as neoplastic, inflammatory, connective, dead or
epithelial. The optional CellViT++ classifier modules and SCOR are not part of this release.

> **Status:** a working local research demo, not for diagnostic use. It segments and classifies
> nuclei in single H&E patches on CPU, from the command line or a browser interface; larger images
> are handled by selecting a crop. It runs locally in Python, with Docker Compose, or from the
> published container image. Its outputs were checked against the upstream CellViT code.

## Setup

Tested on Windows 10 with Python 3.13.14 in PowerShell. You need about 1 GB of free RAM and
an internet connection for the 187 MB checkpoint download.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts\download_checkpoint.py
```

`download_checkpoint.py` fetches `CellViT-256-x40-AMP.pth` from the authors' Zenodo record
into `checkpoints/` and verifies its MD5 and SHA-256.

## Run

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

Open http://127.0.0.1:7860. You can upload an H&E image or press **Load sample patch**.

- **Accepted files:** PNG, JPEG (`.jpg`, `.jpeg`, `.jfif`) or single-page 8-bit TIFF, checked
  by file content rather than file extension. JPEG/TIFF orientation tags are applied. Images
  must be at least 128 px per side and at most 50 megapixels; uploads are capped at 20 MB.
- **Patches of 128–512 px per side** (rectangular is fine) go straight to **Run segmentation**.
- **Larger images** show a preview with a crop box, centred by default. Choose a crop of
  128–512 px per side by clicking two opposite corners on the preview, or by typing x, y, width
  and height (press Enter or move to another field). The crop is cut from the full-resolution
  image, never from the preview. Crops that are too small, too large or past the image edge are
  outlined in red with an explanation, and **Run segmentation** refuses them.
- **Replacing the image:** the upload box keeps showing the current file. Drop another file onto
  it, or click it to choose one; there is no need to press × first. The previous crop, result,
  counts and nucleus details are cleared at once, before the new file is checked. An unreadable or
  unsupported file leaves an empty upload box and interface, with the reason in the status line.
  Loading a new image also cancels a segmentation that is still running for the old one, and ×
  clears everything.
- **How replacement works:** Gradio's single-file box has no drop target once it holds a file, so a
  short script (`JS` in `interface.py`) passes a file dropped on, or chosen from, the occupied box
  to a hidden second upload field, which uploads it normally.
- **Processed patch panel:** shows exactly the pixels sent to the model, with the overlay after a
  run. Changing the crop or loading another image clears the previous result.
- **No resizing:** the patch or crop is padded to a multiple of 16 px for the model, and the
  result is cropped back to the original size.
- **Scale:** the model expects roughly 40× tissue (about 0.25 µm/px). Physical scale is not
  known from the upload, so areas are reported in pixels.

- **Overlay checkbox:** shows or hides the nuclei overlay.
- **Clicking a nucleus:** shows its ID, area and predicted type, plus the clicked position in
  the crop and in the source image.

The model loads once at startup. Toggling the overlay or clicking a nucleus reuses the stored
result; it doesn't run the model again.

## Run the published image

Release v0.1.0 is published as `ghcr.io/mhassan93/histology-explorer:v0.1.0` (linux/amd64). It
needs only Docker, not a clone of this repository. The image contains no model weights, so the
first command downloads and verifies the checkpoint into a Docker volume named
`histology-explorer-checkpoints`. This step runs as root so it can write to the new volume:

```powershell
docker run --rm --user 0:0 -v histology-explorer-checkpoints:/app/checkpoints ghcr.io/mhassan93/histology-explorer:v0.1.0 python scripts/download_checkpoint.py --dir checkpoints
```

Then start the interface, which runs as a non-root user and mounts the volume read-only:

```powershell
docker run --rm --name histology-explorer -p 127.0.0.1:7860:7860 -v histology-explorer-checkpoints:/app/checkpoints:ro ghcr.io/mhassan93/histology-explorer:v0.1.0
```

Once the log shows `[startup] model loaded once`, open **http://localhost:7860**. Stop it with
`docker stop histology-explorer`. The commands are the same in PowerShell and in Linux or macOS
shells. Later runs need only the second command. `docker volume rm histology-explorer-checkpoints`
deletes the downloaded checkpoint.

## Run with Docker

This builds the image from source and runs the same browser interface on CPU inside a container, so no local Python setup is
needed. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (on Windows it
uses WSL 2), start it, then run this from the project folder:

```powershell
docker compose up --build
```

Once the log shows `[startup] model loaded once`, open **http://localhost:7860**. The log also
prints `http://0.0.0.0:7860`; that is the address inside the container, so use `localhost`
instead. The port is only published on this computer (`127.0.0.1`), not on your network.

To stop it, press `Ctrl+C` in that terminal, then run:

```powershell
docker compose down
```

**The three parts:**

| Part | What it is | What it does here |
|---|---|---|
| **Image** `histology-explorer:local` | A read-only package built from the `Dockerfile` | Holds Python, the pinned CPU-only libraries, the app code and the sample patch. It does not hold the model checkpoint. |
| **Containers** `checkpoint` and `app` | Running copies of that image, defined in `compose.yaml` | `checkpoint` downloads and verifies the model file, then exits; it runs as root so it can write into a newly created `checkpoints/` folder. `app` runs the browser interface on port 7860 as a non-root user. `docker compose down` removes both; the image stays. |
| **Mounted folder** `checkpoints/` | The project folder on your computer, shared into the containers at `/app/checkpoints` | Stores the 187 MB checkpoint outside the image, so it is downloaded once and survives rebuilds and `down`. `app` sees it read-only. |

**What the command does:**

1. **Builds the image** `histology-explorer:local` from the `Dockerfile`: Python, the pinned
   CPU-only dependencies, the app code and the sample patch. The first build downloads the
   Python base image and roughly 450 MB of packages; later builds reuse Docker's cache. The image does not contain the model
   checkpoint.
2. **Runs the `checkpoint` service**, a short one-off container. It downloads
   `CellViT-256-x40-AMP.pth` (187 MB) from Zenodo into the project's `checkpoints/` folder if it
   is missing, and verifies its MD5 and SHA-256 either way. It then exits with code 0, which is
   expected.
3. **Starts the `app` service** only if that verification succeeded. `checkpoints/` is mounted
   into it read-only, and the model is loaded once at startup.

The checkpoint therefore lives on your computer in `checkpoints/`, outside the image, and is
downloaded only once. If you already ran `scripts\download_checkpoint.py` for the local setup,
that same file is reused. To force a fresh download, delete the file and start again. On Linux the
downloaded file is owned by root (as is `checkpoints/` if Docker created it), so deleting it
needs `sudo`.

**Running in the background:** `docker compose up --build -d` starts it without tying up the
terminal; `docker compose logs -f app` shows the log, and `docker compose down` stops it. Code
changes need `--build` to take effect.

## Verify

```powershell
.\.venv\Scripts\python.exe scripts\validate_outputs.py --input data\samples\nucls_test_2.png --output-dir outputs\nucls_test_2
.\.venv\Scripts\python.exe scripts\compare_upstream.py --input data\samples\nucls_test_2.png --report outputs\nucls_test_2\upstream_comparison.json
.\.venv\Scripts\python.exe scripts\benchmark.py --input data\samples\nucls_test_2.png --report outputs\nucls_test_2\benchmark.json
```

- **`validate_outputs.py`** checks the checkpoint, the model loading and the outputs' alignment and consistency.
- **`compare_upstream.py`** needs internet access. It compares each stage against pristine upstream code at the pinned commit.
- **`benchmark.py`** measures model loading time, warm inference time and peak process RAM in a fresh process.

## Project layout

```
src/histology_explorer/                browser interface and image input checks
src/histology_explorer/segmentation/   model loading, pipeline, output writing
scripts/                               command-line entry points
third_party/cellvit_inference/         vendored upstream model code (see its README)
data/samples/                          sample patch with provenance
```

## Credits and licences

- **This project's own code** is released under the MIT licence; see [`LICENSE`](LICENSE). That
  covers everything in this repository except the two directories below, which keep their own
  terms. The MIT licence does not extend to the model weights either.
- **Model code and weights:** [CellViT-Inference](https://github.com/TIO-IKIM/CellViT-Inference)
  and [CellViT++](https://github.com/TIO-IKIM/CellViT-plus-plus) by Hörst et al. (IKIM, University
  Medicine Essen), under the upstream
  [`LICENSE`](third_party/cellvit_inference/LICENSE):
  - **SAM-derived components:** Apache 2.0.
  - **HIPT-derived components (code and weights):** Apache 2.0 with the Commons Clause; selling
    them requires written permission from Mahmood Lab.
  - **The CellViT authors' original code:** Apache 2.0 with the Commons Clause; selling it
    requires written permission from Fabian Hörst and Jens Kleesiek, and any use requires citing
    the CellViT and CellViT++ papers.
  - **What the Commons Clause restricts:** the right to "Sell", which the licence defines as
    providing to third parties, for a fee or other consideration (including fees for hosting or
    consulting/support services related to the software), a product or service whose value derives
    entirely or substantially from the software's functionality. It is not a blanket ban on
    commercial use. The per-component summaries at the top of the licence word the restriction
    as "sell, rent, or commercially exploit".
  - **What stays allowed:** everything else Apache 2.0 grants, including use, modification and
    redistribution, on its conditions (licence copy, change notices, retained notices).
  - **Weights:** neither this repository nor its Docker image contains them; they are downloaded
    from the authors' Zenodo record, which states the same terms.
  - **Details:** see [`third_party/cellvit_inference`](third_party/cellvit_inference/README.md)
    for the terms and the citations.
- **Sample patch:** from the NuCLS dataset (Amgad et al., 2022), which its authors release under
  CC0 1.0; see [`data/samples`](data/samples/README.md).
