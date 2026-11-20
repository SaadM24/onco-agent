# ONCO-AGENT — Windows PowerShell runbook

All commands assume the project root is your working directory, `.venv` exists, the source datasets are already extracted, and `onco_agent` is installed with `pip install -e .[dev,dashboard]`. **Do not use PowerShell's global Python unintentionally.**

For convenience in the current PowerShell session:

```powershell
$PY = '.\.venv\Scripts\python.exe'
$NCT_ROOT = 'data\raw\nct\NCT-CRC-HE-100K'  # UPDATE after inspecting actual ZIP layout
$CRC_ROOT = 'data\raw\crc_external\CRC-VAL-HE-7K'  # UPDATE after inspecting ZIP layout
```

Both paths MUST directly contain nine tissue-class subdirectories. Do not proceed until correct.

**1. Integrity audit (reads full data; may consume resources).**

```powershell
& $PY -m onco_agent audit --root $NCT_ROOT --external $CRC_ROOT --output results\data_audit.json
```

Inspect results before continuing. Refuse to proceed when corrupt images, cross-dataset hashes, unexplained classes, or unexpected dimensions indicate a problem.

**2. Split NCT internally.**

```powershell
& $PY -m onco_agent split --root $NCT_ROOT --output results\splits.json --seed 42
```

A patch-level random split is *not* patient-independent. Only set `--groups-csv` when you have actual source-provided mapping of `relative_path,patient_or_slide_id` for **every image**.

**3. Train two different architectures.** Both use internal train images, monitor internal `val`; they never see `policy` or external images during fitting.

```powershell
& $PY -m onco_agent train --manifest results\splits.json --output results\models\resnet18_fp32_v022 --architecture resnet18 --pretrained --batch-size 4 --accumulation 4 --workers 0 --epochs 8 --lr 0.00005 --device cuda
& $PY -m onco_agent train --manifest results\splits.json --output results\models\efficientnet_b0_bn_frozen_full --architecture efficientnet_b0 --pretrained --freeze-bn --batch-size 2 --accumulation 8 --workers 0 --epochs 8 --lr 0.00005 --device cuda
```

If ImageNet weight download is unavailable, omit `--pretrained`, but do not silently equate from-scratch models with transfer-learning experiments.

**4. Score internal validation. Fit temperature here and ONLY here.**

```powershell
& $PY -m onco_agent cache --model results\models\resnet18_fp32_v022\best.pt --manifest results\splits.json --subset val --output results\cache\fast_val.npz --batch-size 4
& $PY -m onco_agent cache --model results\models\efficientnet_b0_bn_frozen_full\best.pt --manifest results\splits.json --subset val --output results\cache\strong_val.npz --batch-size 4
& $PY -m onco_agent calibrate --fast results\cache\fast_val.npz --strong results\cache\strong_val.npz --output results\calibration.json
```

**5. Fit the action policy ONLY on the separate internal `policy` split.**

```powershell
& $PY -m onco_agent cache --model results\models\resnet18_fp32_v022\best.pt --manifest results\splits.json --subset policy --output results\cache\fast_policy.npz --batch-size 4
& $PY -m onco_agent cache --model results\models\efficientnet_b0_bn_frozen_full\best.pt --manifest results\splits.json --subset policy --output results\cache\strong_policy.npz --batch-size 4
& $PY -m onco_agent fit-policy --fast results\cache\fast_policy.npz --strong results\cache\strong_policy.npz --calibration results\calibration.json --output results\learned_policy.json
```

**6. Freeze agent assumptions BEFORE external evaluation.** Defaults are illustrative, not optimized; select them using **internal validation only** and document this choice.

```powershell
& $PY -m onco_agent freeze --calibration results\calibration.json --learned results\learned_policy.json --output results\frozen_config.json --fast-threshold 0.80 --review-threshold 0.65 --blur-edge-threshold 70 --strong-cost 0.10 --review-cost 0.25 --wrong-penalty 2.0
```

**7. Unbiased external evaluation (do not retune after reading results).**

```powershell
& $PY -m onco_agent cache --model results\models\resnet18_fp32_v022\best.pt --external-root $CRC_ROOT --output results\cache\fast_external.npz --batch-size 4
& $PY -m onco_agent cache --model results\models\efficientnet_b0_bn_frozen_full\best.pt --external-root $CRC_ROOT --output results\cache\strong_external.npz --batch-size 4
& $PY -m onco_agent evaluate --fast results\cache\fast_external.npz --strong results\cache\strong_external.npz --config results\frozen_config.json --output results\policy_external --split external
& $PY -m onco_agent plot --metrics results\policy_external\external_metrics.json --output results\policy_external\policy_comparison.png
```

**8. Optional synthetic corruptions (after freezing).**

```powershell
& $PY -m onco_agent stress --model results\models\resnet18_fp32_v022\best.pt --external-root $CRC_ROOT --config results\frozen_config.json --kind blur --output results\policy_external\synthetic_blur.json
```

**9. Live inference on public research patch (not on patients).**

```powershell
& $PY -m onco_agent infer --image data\raw\nct\NCT-CRC-HE-100K\TUM\EXAMPLE.png --fast-model results\models\resnet18_fp32_v022\best.pt --strong-model results\models\efficientnet_b0_bn_frozen_full\best.pt --config results\frozen_config.json --policy learned
& $PY -m streamlit run dashboard/app.py
```

Replace `EXAMPLE.png` with an actual public-dataset filename and the correct extracted folder structure.

**Missing data, GPU unavailable, or training fails?** Run the synthetic checks in README first; record error and environment output. Never invent scores or claim the full research study completed.

## Completed-study note

The models described above were trained locally on an NVIDIA GTX 1650.
EfficientNet-B0 required frozen BatchNorm running statistics during
training. The commands record the completed study; they do not need
to be rerun merely to install the repository.

For exact external metrics, see EXPERIMENT_RESULTS.md.
The external dataset has already been evaluated and must not be
considered an untouched holdout for future tuned versions.

