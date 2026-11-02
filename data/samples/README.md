# Sample patches

## `nucls_test_2.png`

| Property | Value |
|---|---|
| Content | H&E-stained breast cancer tissue, mixed tumour and immune nuclei |
| Dimensions | 256 × 256 px, 8-bit RGB, PNG (no embedded resolution metadata) |
| SHA-256 | `20f7147bec909f1e95f6232b7f44d53eb563691857530694ea11960d53b22352` |
| Obtained from | `test_database/training_database/Example-Segmentation/test/images/test_2.png` in [TIO-IKIM/CellViT-plus-plus](https://github.com/TIO-IKIM/CellViT-plus-plus) at commit `463c5c44bfdebfbe3943597eaa84daf3f5e26a5f` |
| Original dataset | NuCLS (Amgad et al., 2022), built from TCGA breast cancer diagnostic slides |
| Licence | CC0 1.0, as stated in the upstream `readme.txt` and on the [NuCLS website](https://sites.google.com/view/nucls/home) |
| Modifications | None; the file is byte-identical to the upstream copy |

### Image scale

The file carries no scale metadata. The NuCLS paper states that slides were analysed at
40× scan magnification, and TCGA 40× scans are typically about 0.25 µm/px. The CellViT++
authors ship this patch as an example for their 40× models. **Assumption:** the patch is
at about 0.25 µm/px, which matches the training scale of `CellViT-256-x40`. The patch is
used at its native size, with no resizing.

### Citation

Amgad M, Atteya LA, Hussein H, et al. NuCLS: A scalable crowdsourcing approach and dataset
for nucleus classification and segmentation in breast cancer. *GigaScience*. 2022;11:giac037.
https://doi.org/10.1093/gigascience/giac037
