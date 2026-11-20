"""NCT/CRC patch loader and internal development split support.

External CRC-VAL-HE-7K is NEVER accepted as input for training/policy/calibration.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

CLASS_CODES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")


def image_transform(size: int = 224, augment: bool = False):
    ops = [transforms.Resize((size, size))]
    if augment:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90)])
    ops.extend([transforms.ToTensor(), transforms.Normalize(
        mean=[.485, .456, .406], std=[.229, .224, .225])])
    return transforms.Compose(ops)


def load_folder(path, size=224, augment=False) -> ImageFolder:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Unpacked data folder not found: {root}")
    ds = ImageFolder(root=str(root), transform=image_transform(size, augment))
    if not ds.samples:
        raise ValueError(f"No supported images in {root}")
    return ds


def _split(indices, labels, portion, seed, groups=None):
    if groups is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=portion, random_state=seed)
        a, b = next(splitter.split(indices, labels))
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=portion, random_state=seed)
        a, b = next(splitter.split(indices, labels, groups))
    return np.asarray(indices)[a], np.asarray(indices)[b]


def make_internal_splits(ds: ImageFolder, seed=42, policy_fraction=.15,
                         validation_fraction=.15, groups=None):
    """Train, policy-fit, calibration/tuning validation, WITHOUT external images.

    group ids may be supplied only if they are real source patient/slide identifiers.
    A random patch split is NOT patient-independent.
    """
    if not 0 < policy_fraction < 1 or not 0 < validation_fraction < 1 or policy_fraction + validation_fraction >= 1:
        raise ValueError("Invalid split fractions")
    indices = np.arange(len(ds))
    y = np.asarray(ds.targets)
    g = None if groups is None else np.asarray(groups)
    if g is not None and len(g) != len(indices):
        raise ValueError("Group count doesn't match image count")
    train_policy, val = _split(indices, y, validation_fraction, seed, g)
    tr, policy = _split(train_policy, y[train_policy],
                        policy_fraction / (1 - validation_fraction), seed + 1,
                        None if g is None else g[train_policy])
    out = {"train": tr.tolist(), "policy": policy.tolist(), "val": val.tolist()}
    if len(set(tr) | set(policy) | set(val)) != len(ds):
        raise AssertionError("Missing or duplicate image indices")
    if g is not None:
        parts = [set(g[x]) for x in (tr, policy, val)]
        if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError("Patient/slide group overlaps splits")
    return out


def write_splits(ds, splits, dest, seed, source_root=None, groups_provided=False):
    root = Path(source_root or ds.root).resolve()
    rows = []
    for split, indices in splits.items():
        for ix in indices:
            p, y = ds.samples[ix]
            rows.append({"image": Path(p).resolve().relative_to(root).as_posix(),
                         "label": int(y), "subset": split})
    manifest = {"source": str(root), "classes": ds.classes, "seed": seed,
                "group_independent": bool(groups_provided),
                "warning": None if groups_provided else
                "Random patch-level splits may share source patients/slides across partitions.",
                "external_dataset_used_for_selection": False, "rows": rows}
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return hashlib.sha256(dest.read_bytes()).hexdigest()


class ManifestDataset(Dataset):
    """Loads one internal partition or explicit external set; protects held-out use."""
    def __init__(self, manifest_path, subset, augment=False, size=224):
        record = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if subset not in {"train", "policy", "val"}:
            raise ValueError("Only internal train/policy/val are supported")
        self.paths = [(Path(record['source']) / row['image'], int(row['label']))
                      for row in record['rows'] if row['subset'] == subset]
        self.classes = record['classes']
        self.transform = image_transform(size, augment)
        if not self.paths:
            raise ValueError(f"No samples in partition {subset}")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, ix):
        path, label = self.paths[ix]
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        return image, label, str(path)


class FolderDataset(Dataset):
    """Only for external evaluation or inference; never for fitting."""
    def __init__(self, root, classes, size=224):
        ds = load_folder(root, size=size)
        if list(ds.classes) != list(classes):
            raise ValueError(f"Class mismatch! expected {classes}, found {ds.classes}")
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, ix):
        path, label = self.ds.samples[ix]
        image, _ = self.ds[ix]
        return image, label, str(path)
