# Model cards template

For each trained model, create a separate file recording:

- Architecture and pretrained checkpoint origin
- Original dataset and subset manifest SHA-256; exact label mapping
- Best internal validation epoch, hyperparameters, seeds, optimizer, transform and hardware
- Date trained and command used
- Internal and independently held-out performance, explicitly separating the two
- Per-class recall, macro F1, NLL, Brier, ECE and failure examples
- Known limitations including class overlap and sampling bias
- Not-for-clinical-use statement; model weights/checkpoint SHA-256

Do not fabricate any of these values. The repo contains no pre-trained medical models.
