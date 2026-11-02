# Sample patch

`nucls_test_2.png` is a 256 × 256 px H&E patch of breast cancer tissue from the NuCLS dataset,
which was built from TCGA diagnostic slides. It is an unmodified copy of
`test_database/training_database/Example-Segmentation/test/images/test_2.png` in
[TIO-IKIM/CellViT-plus-plus](https://github.com/TIO-IKIM/CellViT-plus-plus) at commit
`463c5c44bfdebfbe3943597eaa84daf3f5e26a5f`.

**Licence:** CC0 1.0, as stated in the upstream `readme.txt` and on the
[NuCLS website](https://sites.google.com/view/nucls/home).

**Scale (assumption):** the file carries no scale metadata. NuCLS slides were analysed at 40×, and
TCGA 40× scans are typically about 0.25 µm/px, so the patch is assumed to be about 0.25 µm/px,
which matches the 40× training scale of `CellViT-256-x40`.

**Citation:** Amgad M, Atteya LA, Hussein H, et al. NuCLS: A scalable crowdsourcing approach and
dataset for nucleus classification and segmentation in breast cancer. *GigaScience*.
2022;11:giac037. https://doi.org/10.1093/gigascience/giac037
