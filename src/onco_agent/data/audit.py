"""Image integrity, duplicated content and provenance checks."""
from __future__ import annotations
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, UnidentifiedImageError

VALID = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def audit_directory(root: str | Path, expected_size: tuple[int, int] | None = None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    classes = Counter()
    hash_paths = defaultdict(list)
    damaged = []
    shape_mismatch = []
    count = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in VALID:
            continue
        count += 1
        classes[p.parent.name] += 1
        rel = str(p.relative_to(root))
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                if expected_size is not None and im.size != expected_size:
                    shape_mismatch.append({"file": rel, "size": list(im.size)})
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            hash_paths[h.hexdigest()].append(rel)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            damaged.append({"file": rel, "error": str(exc)})
    duplicates = [v for v in hash_paths.values() if len(v) > 1]
    return {"root": str(root), "image_count": count, "class_counts": dict(classes),
            "corrupt_files": damaged, "unexpected_shapes": shape_mismatch,
            "duplicate_files_by_sha256": duplicates,
            "sha256_hashes": sorted(hash_paths)}


def cross_dataset_overlap(train_audit: dict, external_audit: dict):
    return sorted(set(train_audit["sha256_hashes"]) & set(external_audit["sha256_hashes"]))
