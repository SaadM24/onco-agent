# Hypothesis and falsification criteria

**Primary question:** Does adaptive tool selection reduce the **error rate on accepted predictions** compared with an ordinary fast classifier and a fixed-abstention baseline, at comparable **coverage and tool-call cost**, under the external colorectal pathology patch distribution?

**H0:** The proposed rule and learned policies do not dominate fixed abstention over comparable coverage or resource budgets. **H1:** An adaptive policy achieves lower selective risk at comparable coverage and budget. Abstention has an explicit review cost and must never be counted as a correct classification.

Pre-register before external evaluation:

- Choose fast and strong architectures; fixed seeds and checkpoints; class mapping, split design, calibrated temperatures; rule thresholds; learned-policy reward costs.
- Compare fast, strong, fixed-abstain, rule-based adaptive, and learned constrained policy on identical samples. Report selective error **alongside coverage**. A low selective error produced by reviewing nearly everything is not automatically useful.
- Report per-class recall and overall macro-F1 for raw and calibrated vision models, as well as ECE, NLL and Brier score. Calibration of max-softmax is not proof of clinical uncertainty calibration.
- If the rule/learned policy is weaker or costs more than a fixed-threshold baseline, state that the hypothesis was **not supported**.
- Count strong-tool invocations; true deployment timings require a separate on-device profiling study.

No tuning on CRC-VAL-HE-7K. Thresholds in the supplied defaults are arbitrary research settings; they should be selected on internal validation and frozen.
