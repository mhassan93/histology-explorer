# Histology Explorer

A browser app for nuclei instance segmentation in H&E histology images, running entirely on CPU.
Upload an image or load the bundled sample, choose a region at full resolution, and inspect each
segmented nucleus with its predicted type. Segmentation uses the published CellViT-256 model and
weights; this project is the application, validation and packaging built around them.

## What this project adds

- **CPU patch inference:** a pipeline around vendored CellViT-256 inference code that needs no GPU.
  It pads each patch to the model's 16 px grid instead of resizing it, then crops the results back
  to the original size.
- **Varied image input:** PNG, JPEG and TIFF uploads are checked by content, not file extension, and
  rotated according to their orientation tags. Unsupported files are rejected with the reason.
- **Original-resolution crop selection:** images larger than 512 px open as a preview with a crop
  box, set by clicking two opposite corners or typing its position and size. The crop is cut from
  the full-resolution pixels, never from the preview.
- **Interactive results:** the tissue type the model predicts for the processed patch or crop,
  nucleus counts by type, a toggleable instance overlay coloured by nucleus type, and the ID, area
  and type of any nucleus you click. Dropping in a new image replaces the old one.
- **Validation against upstream:** a script that compares each stage with the original CellViT code.
- **Docker packaging:** a one-command Compose setup and a published image. The checkpoint is
  downloaded and verified on first run and kept outside the image.

## Input and limits

| | |
|---|---|
| Formats | PNG, JPEG (`.jpg`, `.jpeg`, `.jfif`), single-page 8-bit TIFF |
| Image size | At least 128 px per side and at most 50 megapixels; uploads up to 20 MB |
| Segmented region | 128–512 px per side. Smaller images run directly; larger ones need a crop |
| Scale | Roughly 40× tissue (about 0.25 µm/px). Images are never resized, so areas are in pixels |
| Nucleus types | Neoplastic, inflammatory, connective, dead or epithelial (CellViT-256's PanNuke-trained type head) |
| Tissue type | One of the 19 PanNuke tissue types, from CellViT-256's tissue head, for the processed patch or crop |

## Quick start with Docker Compose

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine with
Compose. Then, from the cloned project folder:

```powershell
docker compose up --build
```

The first run downloads the 187 MB checkpoint from the authors' Zenodo record into `checkpoints/`
and verifies it. When the log shows `[startup] model loaded once`, open **http://localhost:7860**.
Stop with `Ctrl+C`, then `docker compose down`.

## Run the published image

No clone or build is needed. The first command downloads and verifies the checkpoint into a Docker
volume; the second starts the app:

```powershell
docker run --rm --user 0:0 -v histology-explorer-checkpoints:/app/checkpoints ghcr.io/mhassan93/histology-explorer:v0.1.1 python scripts/download_checkpoint.py --dir checkpoints
docker run --rm --name histology-explorer -p 127.0.0.1:7860:7860 -v histology-explorer-checkpoints:/app/checkpoints:ro ghcr.io/mhassan93/histology-explorer:v0.1.1
```

Open the same address, and stop it with `docker stop histology-explorer`. Later runs need only the
second command. The image is built for linux/amd64.

Local Python setup, the command-line tool, the verification scripts and Docker troubleshooting
notes are in [docs/development.md](docs/development.md).

## Validation

On the bundled sample patch, `scripts/compare_upstream.py` runs 26 checks against pristine CellViT
code at the pinned upstream commit. The preprocessed input, all raw network outputs, the final
instance mask and all 46 nucleus records are identical. This shows the app reproduces upstream
inference; it is not an accuracy evaluation against annotated ground truth.

## Credits and licences

- **Model:** This app uses CellViT-256 by Hörst et al. The model and vendored code retain their
  upstream licences; see the [vendored documentation](third_party/cellvit_inference/README.md) for
  terms, attribution and citations.
- **Sample patch:** from the NuCLS dataset (Amgad et al., 2022), released under CC0 1.0; see
  [data/samples](data/samples/README.md).
- **This project's code:** MIT; see [LICENSE](LICENSE). The licence does not cover
  `third_party/`, the sample patch or the model weights.
