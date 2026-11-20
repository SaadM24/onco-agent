# Dataset card and data hygiene

## Original published source

Kather et al., 2018: https://doi.org/10.5281/zenodo.1214456

- **NCT-CRC-HE-100K**: 100,000 H&E colorectal histopathology patches (224×224), nine tissue classes. Source of training, internal validation, and agent-policy-fitting data.
- **CRC-VAL-HE-7K**: 7,180 patches reported to originate from separate patients. Although its filename says VAL, it is the project's **locked external test**.
- **NCT-CRC-HE-100K-NONORM**: optional stain-normalization sensitivity arm. It is related to the other 100K and must **never** be described as independent external data.

Tissue category codes: `ADI` adipose, `BACK` background, `DEB` debris, `LYM` lymphocytes, `MUC` mucus, `MUS` smooth muscle, `NORM` normal colon mucosa, `STR` cancer-associated stroma, `TUM` colorectal adenocarcinoma epithelium. This dataset is not a patient-level clinical outcome cohort.

## Scientific caution

Internal NCT source patient/slide identifiers may not be supplied as usable per-patch metadata. Without verified IDs, code uses a **patch-level** stratified split; correlations across train/validation remain possible, so do not claim patient-independent internal results.

## Required audit

Verify class folders and class names; image count; dimensions; broken files; SHA-256 duplicate images both within sets and across internal/external sets. Retain original ZIP checksums and published attribution. The data are **never bundled** with the code. Treat any human data beyond these public research images as off-limits.

## Distribution limitations

The external cohort differs in patients but not necessarily in geography, clinical procedures or preparation protocols. Synthetic brightness/blur perturbations are not real scanners or prospective clinical shifts. Stain normalization and patch selection can create correlated errors.
