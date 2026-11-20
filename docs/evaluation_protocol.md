# Protocol: information barriers and evaluation

## Sources of supervision

| Partition | Used for | Forbidden uses |
|---|---|---|
| NCT `train` | Fit model weights; validation monitored separately | Claim independent test |
| NCT `val` | Early stopping, temperature calibration, choice of thresholds | Claim unbiased generalisation |
| NCT `policy` | Fit offline cost-sensitive policy (full-information counterfactual actions) | Fit image-model weights |
| CRC external | **Only final independently reported evaluation** | Fit model, calibrator, action policy, thresholds, LLM prompts |

Because early stopping and calibration both inspect `val`, validation metrics are **selection-influenced**. Internal training splits are not patient-independent without genuine group IDs. Run only one decisive external evaluation after freezing, even if external performance is disappointing.

## Policy actions

- **FAST:** always run a lightweight classifier.
- **STRONG:** optionally invoke a second more expensive classifier. Call count is bounded to one per patch.
- **REVIEW:** abstain and explicitly request human inspection; the system never guesses an accepted label on reviewed cases.
- **Rule:** fast confidence and quality flags control second-model invocation; low confidence or model disagreement triggers review.
- **Learned:** predict action values from fast scores and quality statistics with a *supervised cost-sensitive Ridge regression* trained on the internal policy split. This is **not** reinforcement learning and not general multi-agent LLM reasoning.

Labels are present in the policy-fitting reward and final metrics only. They are absent from live inference state features.

## Reporting

Report both coverage (fraction **accepted**) and selective error (among accepted). Abstained predictions get no post-hoc correctness credit. No real tool-call speed-up claim comes from a cache replay; model cost is an assumed scalar tradeoff, to be replaced by separately measured runtime.

Per-image trace logs allow rerunning every policy decision; they include filename, label used for **post-hoc evaluation**, selected action, model calls and accepted prediction. Summaries include macro F1, NLL, multiclass Brier, ECE, per-class precision and recall. ECE alone is a coarse binning approximation.

## Reproducibility

Retain source commit ID, manifest SHA256, dependency versions, trained checkpoint hashes, hardware/driver details, both training histories, calibrated temperatures, policy coefficients, frozen rule thresholds, external results and log files. Report negative or failed experiments. Use different internal seeds for sensitivity if feasible; repeated patches are not equivalent to independent patient samples for confidence intervals.
