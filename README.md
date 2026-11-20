# ONCO-AGENT

**Auditable Multi-Policy Medical Image Research Framework**

> **RESEARCH SOFTWARE ONLY — NOT FOR CLINICAL USE.** This code predicts public-dataset **tissue patch categories**, not cancer diagnoses or treatment outcomes. It must never be used on real patients or private patient imagery.

## Completed experiments

See [Baseline Experimental Results](docs/EXPERIMENT_RESULTS.md) for the completed internal and external evaluation, paired-error analysis, and limitations.

## The research question

Can a tool-selecting classifier route ambiguous histopathology patches to additional analysis or human review to reduce accepted-prediction errors while respecting inference costs—compared with single-model inference and fixed abstention? We make no claim of medical validity or agent improvement without real experiments.

### What is implemented

- NCT class-folder dataset loader; stratified **train / policy / validation** internal partitions; optional *verified* patient/slide group splits; image integrity and duplicate-hash auditing.
- Lightweight **ResNet18** and contrasting **EfficientNet-B0** image models, optional ImageNet pretrained weights, PyTorch training, best checkpoints, seed and run metadata; CPU/CUDA.
- Cached per-image logits with quality statistics, guarded alignment checks and optional external CRC-VAL-HE-7K evaluation.
- Validation-only temperature calibration; full-information cost-sensitive **learned** action selection (Ridge action values) trained only on the policy split. **Not RL, not an LLM agent.** Also baseline-only, fixed-threshold and rule-based adaptive tool policies.
- Five policy comparisons with selective error, coverage, strong tool invocations and modeled extra cost. Classifier metrics: macro-F1, per-class precision/recall, NLL, Brier, ECE; before/after calibration.
- Synthetic blur, brightness and color stress tests (explicitly *not* independent datasets).
- Auditable CSV traces, JSON metrics, a summary plot, research-only Streamlit dashboard and automated tests.

### Critical limitations

1. **Datasets and trained checkpoints are not distributed.** Real-data experiments were completed locally; see [experimental results](docs/EXPERIMENT_RESULTS.md) for performance and limitations.
2. Source record filenames can have nested directories; point `--root` to the directory containing the **nine immediate class-code folders**.
3. Unless genuine group IDs are provided, internal splits are **patch-level** and may share slide/patient origins. CRC-VAL-HE-7K is a separate-patient external set according to the dataset authors.
4. Calibration and policy thresholds based on internal validation are selection-influenced. Treat **external evaluation as held out** and do not retune after viewing it.
5. Learned policy = supervised cost-sensitive action-value regression on full-information examples; **do not call it a reinforcement-learning agent**. The rule policy selects tools based on state, while a genuine LLM tool-use benchmark remains future work.
6. Models use the same underlying tissue domain and pretrained ImageNet representations. Errors can correlate; no guaranteed reliability improvement.
7. True per-sample adaptive deployment latency must be measured separately. Cache replay measures **tool-call counts and arbitrary unit costs**, not saved wall-clock compute.
8. Test artifacts and synthetic-data experiments are not publication-quality empirical evidence.

## 1. Windows setup (PowerShell, recommended Python 3.11)

Unzip the project and open PowerShell in the **`onco-agent`** directory. Use your existing Python 3.11 environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m onco_agent --help
```

If you want GPU training, check your *actual* installed GPU and CUDA support:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If `False`, see the current official [PyTorch Windows installation instructions](https://pytorch.org/get-started/locally/) for a CUDA-enabled `torch`/`torchvision` pairing compatible with your NVIDIA driver. Don't randomly mix PyTorch CUDA wheel versions. CPU mode remains available.

**GTX 1650 (4 GB) / 8 GB RAM:** use `--batch-size 4 --workers 0 --accumulation 4`, and close memory-heavy programs. Pretrained weights need internet access *once*. Large data downloads and extracted images require substantial additional free disk space.

### Synthetic end-to-end installation check (NOT real research)

```powershell
.\.venv\Scripts\python.exe -m onco_agent synthetic-data --output data/synthetic_smoke --count 16 --size 64
.\.venv\Scripts\python.exe -m onco_agent split --root data/synthetic_smoke --output results/smoke_splits.json --image-size 64
.\.venv\Scripts\python.exe -m onco_agent train --manifest results/smoke_splits.json --output results/smoke_fast --architecture tiny_cnn --epochs 1 --size 64 --batch-size 8 --device cpu
```

This validates code and execution only. Training on artificial color patches **must never** be described as histopathology research.

## 2. Obtain the actual datasets

Original record: **[Kather et al. (2018), Zenodo 10.5281/zenodo.1214456](https://zenodo.org/records/1214456)**. Verify current license and redistribution conditions from the original record before any publication or distribution. Original source is widely listed as **CC BY 4.0**; give attribution.

Download **from Zenodo** (do not put large ZIPs into GitHub):

| File | Purpose | Recorded MD5 |
|---|---|---|
| `NCT-CRC-HE-100K.zip` | Model development; normalized NCT patches | `6fd702d11df6292bc054397ae038a464` |
| `CRC-VAL-HE-7K.zip` | **Independent external testing only**, never training or tuning | `2fd1651b4f94ebd818ebf90ad2b6ce06` |
| `NCT-CRC-HE-100K-NONORM.zip` | Optional stain-shift sensitivity arm; **not independent** of normalized NCT | `035777cf327776a71a05c95da6d6325f` |

The data record lists the normalized 100K archive at approximately 11.7 GB and the external archive at 800 MB; extracted files and checkpoints need *additional* space.

Download using your browser into `data/archives/`, then extract safely with checksum validation:

```powershell
.\.venv\Scripts\python.exe -m onco_agent extract --archive data/archives/NCT-CRC-HE-100K.zip --dest data/raw/nct
.\.venv\Scripts\python.exe -m onco_agent extract --archive data/archives/CRC-VAL-HE-7K.zip --dest data/raw/crc_external
```

**After extraction**, inspect directory levels. In commands below, `NCT_ROOT` and `CRC_ROOT` mean the folders whose direct subfolders are `ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, `TUM`. Do not accidentally point to a parent container.

## 3. Execute the experiment in the correct order

You can follow [docs/WINDOWS_COMMANDS.md](docs/WINDOWS_COMMANDS.md) for exact full commands. Quick summary:

1. `audit` the internal and external data for images, dimensions, hashes and overlaps; save report.
2. `split` **only the NCT training dataset**. The resulting manifest contains **train**, **policy** and **val** partitions and a transparent patient-group independence warning.
3. `train` ResNet18 and EfficientNet-B0 using `--pretrained` if internet access allows. Save separate checkpoints.
4. `cache` both models on `val`, then `calibrate` only on val.
5. `cache` both on `policy`; `fit-policy` on policy images and their labels, using fixed calibration.
6. `freeze` calibration, rule thresholds, policy and cost assumptions **before accessing external labels**.
7. `cache` both on independent external patches, then `evaluate --split external` to create metrics/traces. Never adjust based on this test.
8. `plot` summary; optionally `stress` on synthetic image corruptions; inspect dashboard.

Do **not** use external sample images for architecture selection, threshold tuning, prompts or learned-policy fitting.

## 4. Dashboard (after training)

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Fill in paths to trained models and frozen config. The dashboard only displays locally loaded model output and transparent tool traces; it does not offer medical advice.

## 5. Files

- `src/onco_agent/data/` integrity checks, transforms and splitting
- `src/onco_agent/models/` model factory, trainer and checkpoints
- `src/onco_agent/calibration/` validation-only temperature fitting
- `src/onco_agent/agents/` baseline/rule policies and offline learned policy
- `src/onco_agent/evaluation/` caches, metrics, replay, external & stress tests
- `src/onco_agent/reporting/` reproducible visualisation
- `dashboard/app.py` optional Streamlit viewer
- `tests/` unit and integration tests
- `docs/` research scope, constraints and reproducibility protocol
- `IMPLEMENTATION_STATUS.md` honest implementation-vs-results checklist

## Attribution and research transparency

Kather, J. N., Halama, N., & Marx, A. (2018). *100,000 histological images of human colorectal cancer and healthy tissue* (v0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1214456

This source-code package is distributed without model weights or data. Local dataset licenses still govern the user’s research use. If you publish or submit work based on this software, document any AI assistance transparently according to the receiving institution's requirements. Independently review, understand and validate the code before making claims of authorship or results.

