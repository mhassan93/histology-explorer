# Vendored subset of CellViT-Inference

This directory contains the minimum set of upstream files needed to build the CellViT-256
model and run its CPU/NumPy instance postprocessing on a single image patch.

| | |
|---|---|
| Upstream | https://github.com/TIO-IKIM/CellViT-Inference |
| Commit | `d2143ded642846c6265fcfda9711a0376a894d8e` (2025-08-13, package version 1.0.9) |
| Licence | See [`LICENSE`](LICENSE): Apache 2.0 for the SAM-derived components. Apache 2.0 with the Commons Clause for the HIPT-derived components (no right to "Sell", as defined in Appendix B of the licence, without written permission from Mahmood Lab) and for the original CellViT code (the same, with permission from Fabian Hörst and Jens Kleesiek, plus mandatory citation). The repository's top-level MIT licence does not apply to this directory. |
| Authors | Fabian Hörst, Jens Kleesiek and colleagues, Institute for AI in Medicine (IKIM), University Medicine Essen |

## Files

All files are byte-identical to upstream, which `scripts/compare_upstream.py` checks against
the git blob hashes, except for:

- `cellvit/inference/postprocessing_numpy.py`: removed `import ray` and the import of
  `cellvit.data.dataclass.wsi`, and turned the `WSI`/`WSIMetadata` type hints into string
  annotations. Ray is only used by the WSI orchestration (`create_batch_pooling_actor`),
  which this project does not call. The postprocessing algorithm is unchanged. A
  modification notice is included at the top of the file, as Apache 2.0 §4(b) requires.

## Required citations

The upstream licence requires these citations whenever any part of the work is used:

Hörst, F., Rempe, M., Heine, L., Seibold, C., Keyl, J., Baldini, G., Ugurel, S., Siveke, J.,
Grünwald, B., Egger, J., & Kleesiek, J. (2023). CellViT: Vision Transformers for precise cell
segmentation and classification. https://doi.org/10.48550/ARXIV.2306.15350

Hörst, F., Rempe, M., Becker, H., Heine, L., Keyl, J., & Kleesiek, J. (2025). CellViT++:
Energy-Efficient and Adaptive Cell Segmentation and Classification Using Foundation Models
(Version 1). arXiv. https://doi.org/10.48550/ARXIV.2501.05269

The HIPT ViT-256 encoder is derived from Chen, R. J., et al. (2022). Scaling Vision
Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning. *CVPR*.
https://github.com/mahmoodlab/HIPT
