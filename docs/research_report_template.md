# ONCO-AGENT Research Report — TEMPLATE (not results)

**Status: NO REAL EXPERIMENTAL RESULTS INCLUDED.**

## Abstract
[Write only after final evaluation; no simulated scores.]

## Research question
Can constrained adaptive model/tool selection reduce accepted-error at comparable coverage and action cost relative to fixed abstention?

## Data and split integrity
- Original source and license: Kather et al. (2018), Zenodo DOI 10.5281/zenodo.1214456.
- Manifest SHA-256: [fill]
- Group-independent internal split? [yes/no, with proof]
- External data duplicates/overlap? [audit numbers]

## Methods
- Fast model: [architecture, training, selection]
- Strong model: [architecture, training, selection]
- Temperature fit: internal validation only [temperatures]
- Policies: fast only / strong only / fixed abstention / rule / supervised learned action values
- Frozen selection and cost parameters: [full config checksum]

## Results — fill from actual output only

| Policy | Coverage | Selective error | Strong calls/image | Modeled extra cost/image |
|---|---:|---:|---:|---:|
| Fast | [ ] | [ ] | [ ] | [ ] |
| Strong | [ ] | [ ] | [ ] | [ ] |
| Fixed abstention | [ ] | [ ] | [ ] | [ ] |
| Rule | [ ] | [ ] | [ ] | [ ] |
| Learned | [ ] | [ ] | [ ] | [ ] |

Vision-model metrics: report raw/calibrated macro-F1, accuracy, per-class recall, NLL, Brier and ECE on independent external test.

## Findings / negative results
[Interpret evidence; do not imply medical utility.]

## Limitations
[See `docs/limitations.md`; give concrete error cases from trace CSV.]

## Computational reproducibility
[Commit hash, environment output, CUDA/drivers, batch size, training command, archive hashes, saved manifests and checkpoint hashes.]
