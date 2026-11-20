"""BatchNorm freeze regression tests for small-microbatch transfer learning."""
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from onco_agent.models.training import one_epoch


class FakeSet(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.rand(3, 8, 8), i % 2, f"fake_{i}"


class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.bn(x).mean(dim=(2, 3)))


def test_frozen_bn_does_not_update_running_statistics():
    torch.manual_seed(42)
    net = BNNet()
    before = net.bn.running_mean.detach().clone()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    result = one_epoch(net, DataLoader(FakeSet(), batch_size=2), 'cpu', opt, freeze_bn=True)
    assert result['samples'] == 8
    assert torch.equal(before, net.bn.running_mean)
    assert net.bn.training is False
    assert net.fc.training is True


def test_default_bn_updates_running_statistics():
    torch.manual_seed(42)
    net = BNNet()
    before = net.bn.running_mean.detach().clone()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    result = one_epoch(net, DataLoader(FakeSet(), batch_size=2), 'cpu', opt)
    assert result['samples'] == 8
    assert not torch.equal(before, net.bn.running_mean)
