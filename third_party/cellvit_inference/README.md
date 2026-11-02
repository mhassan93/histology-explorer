# Vendored CellViT-Inference code

The CellViT-256 model code and its CPU/NumPy instance postprocessing are taken from
[TIO-IKIM/CellViT-Inference](https://github.com/TIO-IKIM/CellViT-Inference) at commit
`d2143ded642846c6265fcfda9711a0376a894d8e` (version 1.0.9), by Fabian Hörst, Jens Kleesiek and
colleagues. Only the files this app needs are included. They remain under the upstream
[LICENSE](LICENSE), which sets out the terms, attribution and required citations; the project's
top-level MIT licence does not apply to this directory.

## Adapted file

`cellvit/inference/postprocessing_numpy.py` drops two imports used only for whole-slide processing
(`ray` and `cellvit.data.dataclass.wsi`) and quotes the `WSI`/`WSIMetadata` type hints. The
postprocessing algorithm is unchanged, and the file carries a modification notice. All other files
are byte-identical to upstream.
