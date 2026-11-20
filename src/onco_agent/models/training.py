"""Memory-conscious supervised learning, independent of external data."""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from onco_agent.data.datasets import ManifestDataset
from onco_agent.models.factory import create_model


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(request):
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or install CUDA PyTorch.")
    return request


def _check_finite(value, *, field, phase, step, paths):
    """Fail early rather than log NaN after an entire expensive epoch."""
    if not bool(torch.isfinite(value).all().item()):
        preview = [str(p) for p in paths[:3]]
        raise FloatingPointError(
            f"Non-finite {field} during {phase} at batch {step + 1}; "
            f"example paths={preview}. No reliable training result was produced."
        )


def one_epoch(model, loader, device, optimizer=None, accum=1, amp=False, grad_clip=1.0, freeze_bn=False):
    """One supervised epoch with finite checks and opt-in mixed precision."""
    training = optimizer is not None
    if accum < 1:
        raise ValueError("accum must be >= 1")
    model.train(training)
    # With a microbatch of 2, EfficientNet BatchNorm statistics can drift heavily.
    # In freeze mode, pretrained running means/variances are retained while
    # affine parameters (gamma/beta) remain trainable.
    if training and freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp and device == 'cuda')
    if training:
        optimizer.zero_grad(set_to_none=True)
    total_loss, correct, count = 0., 0, 0
    phase = 'train' if training else 'validation'
    for step, (image, label, paths) in enumerate(loader):
        _check_finite(image, field='input images', phase=phase, step=step, paths=paths)
        image, label = image.to(device), label.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu',
                                enabled=amp and device == 'cuda'):
                logits = model(image)
                _check_finite(logits, field='logits', phase=phase, step=step, paths=paths)
                loss = criterion(logits.float(), label)
        _check_finite(loss, field='loss', phase=phase, step=step, paths=paths)
        if training:
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                # The gradient can be temporarily large; clipping protects updates.
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip, error_if_nonfinite=True
                    )
                except RuntimeError as exc:
                    raise FloatingPointError(
                        f"Non-finite gradient during train at batch {step + 1}; "
                        f"example paths={[str(p) for p in paths[:3]]}"
                    ) from exc
                if not bool(torch.isfinite(grad_norm).item()):
                    raise FloatingPointError(f"Non-finite gradient norm at batch {step + 1}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # Periodically check weights; batch-level loss/logit checks catch earlier failures.
                if (step + 1) % (50 * accum) == 0 or step + 1 == len(loader):
                    if any(not bool(torch.isfinite(p).all().item()) for p in model.parameters()):
                        raise FloatingPointError(f"Non-finite model weights after train batch {step + 1}")
        total_loss += float(loss.detach().item()) * len(label)
        correct += int((logits.argmax(1) == label).sum().item())
        count += len(label)
    return {'loss': total_loss / count, 'accuracy': correct / count, 'samples': count}


def train(manifest, output, architecture='resnet18', epochs=8, batch_size=8,
          lr=1e-4, size=224, device='auto', pretrained=False, workers=0, seed=42,
          accumulation=1, patience=4, amp=False, grad_clip=1.0, freeze_bn=False):
    """Only model development train/val partitions. Hold out policy fit and external test."""
    if freeze_bn and not pretrained:
        raise ValueError('Frozen BatchNorm requires --pretrained to retain meaningful running statistics.')
    set_seed(seed)
    device = choose_device(device)
    train_ds = ManifestDataset(manifest, 'train', augment=True, size=size)
    val_ds = ManifestDataset(manifest, 'val', augment=False, size=size)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=device == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    model = create_model(architecture, len(train_ds.classes), pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float('inf'); bad = 0
    for epoch in range(1, epochs+1):
        tic = time.perf_counter()
        try:
            tr = one_epoch(model, loader, device, opt, accumulation, amp=amp, grad_clip=grad_clip, freeze_bn=freeze_bn)
            va = one_epoch(model, val_loader, device)
        except FloatingPointError as exc:
            (output / 'failure.json').write_text(json.dumps({
                'epoch': epoch, 'error': str(exc), 'amp': amp, 'lr': lr, 'freeze_bn': freeze_bn,
                'disclaimer': 'Failed numerical-safety check; this run is invalid.'
            }, indent=2), encoding='utf-8')
            raise
        line = {'epoch': epoch, 'train': tr, 'val': va, 'seconds': round(time.perf_counter()-tic, 2)}
        history.append(line)
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(line), flush=True)
        if va['loss'] < best:
            best, bad = va['loss'], 0
            torch.save({'architecture': architecture, 'num_classes': len(train_ds.classes),
                        'classes': train_ds.classes, 'size': size, 'seed': seed,
                        'pretrained': pretrained, 'state_dict': model.state_dict(),
                        'epoch': epoch, 'validation_loss': best}, output / 'best.pt')
        else:
            bad += 1
            if bad >= patience:
                break
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (output / 'experiment.json').write_text(json.dumps({
        'architecture': architecture, 'manifest': str(Path(manifest).resolve()),
        'epochs_requested': epochs, 'epochs_completed': len(history), 'seed': seed,
        'device': device, 'pretrained': pretrained, 'batch_size': batch_size, 'lr': lr,
        'size': size, 'accumulation': accumulation, 'amp': amp, 'grad_clip': grad_clip,
        'freeze_bn': freeze_bn,
        'disclaimer': 'No external CRC-VAL-HE-7K used in training or selection.'}, indent=2), encoding='utf-8')
    return history
