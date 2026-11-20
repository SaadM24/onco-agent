"""Synthetic stress tests AFTER freezing the protocol; never call them external cohorts."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from scipy.special import softmax
from onco_agent.models.factory import load_checkpoint
from onco_agent.models.training import choose_device
from onco_agent.data.datasets import image_transform, load_folder
from onco_agent.evaluation.metrics import classification_metrics
from onco_agent.evaluation.robustness import perturb


def stress_test(checkpoint, external_root, config, perturbation, output,
                device='auto', max_items=None):
    dev=choose_device(device)
    model, meta=load_checkpoint(checkpoint,dev)
    dataset=load_folder(external_root)
    if dataset.classes != meta['classes']:
        raise ValueError('Class mismatch')
    cfg=json.loads(Path(config).read_text(encoding='utf-8'))
    # Only a single checkpoint, and no fit to external labels.
    if 'calibration' not in cfg or not cfg.get('protocol_frozen_before_external'):
        raise ValueError('Freeze protocol before stress test')
    fast_name=Path(checkpoint).resolve()
    fast_cache=cfg.get('fast_checkpoint_sha256')
    temp=cfg['calibration']['fast_temperature']
    count=min(max_items or len(dataset),len(dataset))
    probs=[];labels=[]
    with torch.inference_mode():
        for path,label in dataset.samples[:count]:
            with Image.open(path) as im:
                modified=perturb(im.convert('RGB'),perturbation)
                tensor=image_transform(meta['size'])(modified).unsqueeze(0).to(dev)
            probs.append(softmax(model(tensor).cpu().numpy()[0]/temp))
            labels.append(label)
    values={'dataset':'CRC-VAL-HE-7K or supplied external class-folder dataset',
            'evaluation_type':'synthetic_corruption_of_external_patches',
            'independent_external_dataset':False,
            'corruption':perturbation,'samples_tested':count,
            'subset_or_full':'subset' if count < len(dataset) else 'full',
            'checkpoint':str(fast_name),'metrics':classification_metrics(np.asarray(probs),np.asarray(labels))}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(values,indent=2),encoding='utf-8')
    return values
