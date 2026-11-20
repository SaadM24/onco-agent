"""Reusable model prediction caches. Offline replay does NOT estimate true deployment latency."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from onco_agent.data.datasets import ManifestDataset, FolderDataset
from onco_agent.models.factory import load_checkpoint
from onco_agent.models.training import choose_device
from onco_agent.tools.image_quality import quality_indicators


def _quality(path):
    with Image.open(path) as im:
        rgb = im.convert('RGB').resize((64, 64))
        return quality_indicators(np.asarray(rgb))


def inference_cache(model_path, dest, *, manifest=None, subset=None, external_root=None,
                    batch_size=16, workers=0, device='auto', max_items=None):
    if bool(manifest) == bool(external_root):
        raise ValueError('Specify either an internal --manifest or --external-root')
    if manifest and subset not in {'policy', 'val'}:
        raise ValueError('Caches for internal training images not allowed; use policy or val')
    device = choose_device(device)
    model, ckpt = load_checkpoint(model_path, device)
    if manifest:
        ds = ManifestDataset(manifest, subset, size=ckpt['size'])
    else:
        ds = FolderDataset(external_root, ckpt['classes'], size=ckpt['size'])
    total_available = len(ds)
    if max_items is not None:
        from torch.utils.data import Subset
        if max_items < 1:
            raise ValueError('--max-items must be positive')
        ds = Subset(ds, range(min(max_items, len(ds))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    rows, labels, logits, q, total_time = [], [], [], [], 0.0
    with torch.inference_mode():
        for batch, y, paths in loader:
            tic = time.perf_counter()
            z = model(batch.to(device)).cpu().numpy()
            total_time += time.perf_counter() - tic
            logits.extend(z); labels.extend(y.numpy().tolist()); rows.extend(paths)
            q.extend([_quality(path) for path in paths])
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, ids=np.asarray(rows), labels=np.asarray(labels, dtype=int),
                        logits=np.asarray(logits, dtype=np.float32),
                        brightness=np.asarray([x['mean_brightness'] for x in q]),
                        edge_variance=np.asarray([x['edge_variance'] for x in q]),
                        saturation=np.asarray([x['saturation_fraction'] for x in q]))
    dest.with_suffix('.json').write_text(json.dumps({
        'checkpoint': str(Path(model_path).resolve()), 'source_kind': 'external' if external_root else subset,
        'n_samples': len(labels), 'n_available': total_available,
        'incomplete_subset': len(labels) < total_available, 'classes': ckpt['classes'], 'device': device,
        'measured_model_forward_seconds_total': total_time,
        'warning': 'Timing excludes image I/O; offline replay cannot measure adaptive runtime.'}, indent=2), encoding='utf-8')
    return dest


def paired_caches(a, b):
    x, y = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    if not np.array_equal(x['ids'], y['ids']) or not np.array_equal(x['labels'], y['labels']):
        raise ValueError('Cache images and labels differ or have a different order')
    meta_a = json.loads(Path(a).with_suffix('.json').read_text(encoding='utf-8'))
    meta_b = json.loads(Path(b).with_suffix('.json').read_text(encoding='utf-8'))
    if meta_a['classes'] != meta_b['classes']:
        raise ValueError('Model class mappings differ between caches')
    if x['logits'].shape != y['logits'].shape:
        raise ValueError('Model classes or sample numbers differ')
    return x, y
