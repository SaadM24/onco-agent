# ONCO-AGENT — Implementation and Validation Status

Research software for nine-class colorectal histopathology patch classification,
confidence calibration and adaptive model selection. Not for clinical use.

## Completed experiments

| Component | Status |
|---|---|
| NCT-CRC-HE-100K integrity audit | 100,000 images; zero corrupt images |
| CRC-VAL-HE-7K external audit | 7,180 images; zero corrupt images |
| Cross-dataset exact-hash overlap | Zero detected |
| Internal partitioning | 69,999 train; 15,001 policy; 15,000 validation |
| ResNet18 training | Completed, 8 epochs |
| EfficientNet-B0 training | Completed, 8 epochs; BatchNorm frozen |
| Temperature scaling | Completed on internal validation |
| Learned cost-sensitive routing | Fitted on 15,001 policy images |
| Frozen policy evaluation | Completed on external dataset |
| Automated tests | 15 passed locally |

## External results

| System | Accuracy | Strong-model invocation |
|---|---:|---:|
| ResNet18 | 89.57% | 0% |
| EfficientNet-B0 | 96.66% | 100% |
| Learned router | 96.28% | 25.78% |

External macro-F1:
- ResNet18: 0.8443
- EfficientNet-B0: 0.9546

The learned router corrected 540 of 594 errors for which
EfficientNet was correct and ResNet18 was wrong.

It did not outperform standalone EfficientNet-B0 externally.

## Implemented but not fully empirically validated

- Synthetic image-corruption stress-testing functionality
- Streamlit dashboard
- Live sequential inference and actual inference-latency savings

## Not implemented

- Reinforcement-learning controller
- LLM-driven planning agent
- Full multimodal clinical reasoning
- Validated clinical decision support

## Research limitations

Internal random partitions may share patient or slide origins.
External cohort separation follows the dataset authors' provenance.
Calibration and model selection used internal development data.
The learned controller did not abstain in external evaluation.
Model-call reductions are simulated, not measured runtime speedups.
The external dataset has been examined and is no longer an
untouched test set for subsequent redesigned methods.

Datasets, checkpoints, caches and full local experiment artifacts
are excluded from this source repository.

See docs/EXPERIMENT_RESULTS.md for detailed results.
