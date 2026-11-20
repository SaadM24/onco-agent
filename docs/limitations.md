# Threats to validity

1. Patch-level supervised labels cannot establish any diagnosis, patient risk, prognosis or treatment recommendation.
2. External CRC-VAL-HE-7K is separate-patient according to its originator, but not necessarily separate research centre/collection pipeline.
3. Small validation sample subsets and random patch splits can produce unstable estimates and accidental slide correlation.
4. Accuracy and macro-F1 do not reflect the asymmetric costs of clinical error. This software is NOT clinically validated.
5. The agent's decision signal (softmax scores and simple pixel-quality statistics) may be poorly calibrated under distribution shift.
6. The learned policy is a fully supervised full-information action-value estimator, **not** RL, an autonomous LLM researcher, or clinical decision support.
7. Model costs are unitless, not real milliseconds or monetary charges. Offline cache generation evaluates all models, even those not invoked by the replay policy.
8. No explainable saliency localisation or segmentation benchmark is included; a pathology specialist cannot infer tissue mechanisms from an action trace alone.
9. External metrics should be regarded as a single finite collection of patches; they do not constitute deployment validation or patient-level outcomes.
10. No raw data, model checkpoints, private patient images or actual medical prediction metrics ship with this repository.
