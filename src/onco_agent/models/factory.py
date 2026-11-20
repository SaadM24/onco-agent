"""Reference architectures with explicit ImageNet transfer learning."""
import torch
from torch import nn
from torchvision.models import (resnet18, efficientnet_b0, ResNet18_Weights,
                                EfficientNet_B0_Weights)


class TinyConvNet(nn.Module):
    """Only for offline tests; NOT a scientifically meaningful research baseline."""
    def __init__(self, num_classes=9):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1),
                                 nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(16, num_classes))
    def forward(self, x):
        return self.net(x)


def create_model(name, num_classes, pretrained=False):
    if name == "tiny_cnn":
        if pretrained:
            raise ValueError("Tiny model has no pretrained weights")
        return TinyConvNet(num_classes)
    if name == "resnet18":
        return _head_resnet(num_classes) if pretrained else resnet18(weights=None, num_classes=num_classes)
    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {name}")


def _head_resnet(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path, device="cpu"):
    # Load only self-created / trusted checkpoints. weights_only avoids arbitrary pickle execution.
    package = torch.load(path, map_location=device, weights_only=True)
    model = create_model(package["architecture"], package["num_classes"])
    model.load_state_dict(package["state_dict"])
    model.eval().to(device)
    return model, package
