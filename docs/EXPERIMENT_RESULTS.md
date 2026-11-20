# ONCO-AGENT — Baseline Experimental Results

## Scope

Research-only nine-class colorectal histopathology patch classification
with confidence calibration and adaptive model routing.

Dataset: 100,000 NCT-CRC-HE-100K patches.
Internal partitions: 69,999 train, 15,001 policy, 15,000 validation.
External evaluation: 7,180 CRC-VAL-HE-7K patches.
No exact cross-dataset duplicate hashes were detected.

Dataset source: https://zenodo.org/records/1214456

## Standalone classifier performance

| Model | Internal validation accuracy | External accuracy | External macro-F1 |
|---|---:|---:|---:|
| ResNet18 | 98.84% | 89.57% | 0.8443 |
| EfficientNet-B0 | 99.48% | 96.66% | 0.9546 |

## External decision-policy comparison

| Policy | Coverage | Accepted accuracy | Strong-model calls |
|---|---:|---:|---:|
| ResNet18 only | 100% | 89.57% | 0% |
| EfficientNet-B0 only | 100% | 96.66% | 100% |
| Fixed abstention | 86.92% | 93.96% | 0% |
| Rule-based routing | 94.30% | 94.24% | 23.05% |
| Learned routing | 100% | 96.28% | 25.78% |

The learned controller reduced strong-model invocations but did
not outperform EfficientNet-B0 on external prediction accuracy.

## Paired external failure analysis

- Both models correct: 6,346 images.
- ResNet18 wrong, EfficientNet correct: 594 images.
- ResNet18 correct, EfficientNet wrong: 85 images.
- Both models wrong: 155 images.
- The controller routed 540 of 594 correctable ResNet18 errors.
- It missed 54 available corrections and made 58 harmful switches.
- Total learned-controller errors: 267.

## Scientific limitations

- Internal image-patch splits are not verified patient-independent.
- External patient separation follows the dataset authors' provenance.
- Internal calibration and model selection share development data.
- The learned controller did not abstain during external evaluation.
- Model-call reductions are simulated; runtime savings are unverified.
- The controller is supervised cost-sensitive routing, not RL or an LLM agent.
- These tissue-patch classifications are not clinical diagnoses.
- External data have now been examined and must not be reused as
  an untouched test set for subsequently tuned methods.

Datasets and trained checkpoints are excluded from the repository.
Experiments were conducted locally; source and reproduction
instructions are provided separately.
