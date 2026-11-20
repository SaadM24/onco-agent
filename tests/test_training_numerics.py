"""Numerical-safety regression tests for long-running model training."""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from onco_agent.models.training import one_epoch


class DemoDataset(Dataset):
    def __init__(self, bad=False):
        self.bad = bad
    def __len__(self):
        return 4
    def __getitem__(self, i):
        x = torch.full((3, 8, 8), float(i) / 4)
        if self.bad and i == 0:
            x[0, 0, 0] = float("nan")
        return x, i % 2, f"demo_{i}.png"


def model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))


def test_nonfinite_input_fails_early():
    with pytest.raises(FloatingPointError, match="input images"):
        one_epoch(model(), DataLoader(DemoDataset(bad=True), batch_size=2), "cpu")


def test_finite_cpu_train_and_validate():
    net = model()
    loader = DataLoader(DemoDataset(), batch_size=2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    tr = one_epoch(net, loader, "cpu", optimizer=opt, accum=2, amp=False)
    val = one_epoch(net, loader, "cpu")
    assert torch.isfinite(torch.tensor(tr['loss']))
    assert torch.isfinite(torch.tensor(val['loss']))
    assert tr['samples'] == val['samples'] == 4
