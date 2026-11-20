"""Command-line entry point. Use `onco-agent --help` or python -m onco_agent."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from PIL import Image
import numpy as np


def dump(obj):
    print(json.dumps(obj, indent=2, default=str))


def split_command(args):
    from onco_agent.data.datasets import load_folder, make_internal_splits, write_splits
    ds = load_folder(args.root, args.image_size)
    groups = None
    if args.groups_csv:
        mapping={}
        with open(args.groups_csv, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                mapping[row['relative_path'].replace('\\','/')] = row['patient_or_slide_id']
        groups = []
        for path,_ in ds.samples:
            rel=Path(path).relative_to(ds.root).as_posix()
            if rel not in mapping or not mapping[rel]:
                raise ValueError(f'Missing independently verified group ID for {rel}')
            groups.append(mapping[rel])
    splits=make_internal_splits(ds,seed=args.seed,policy_fraction=args.policy_fraction,
                                validation_fraction=args.val_fraction,groups=groups)
    h=write_splits(ds,splits,args.output,args.seed,groups_provided=groups is not None)
    dump({'classes':ds.classes,'sizes':{k:len(v) for k,v in splits.items()},
          'manifest':args.output,'manifest_sha256':h,
          'group_independent':groups is not None,
          'warning':None if groups else 'Patch-level split; not independently grouped by patient or slide.'})


def audit_command(args):
    from onco_agent.data.audit import audit_directory, cross_dataset_overlap
    result=audit_directory(args.root,expected_size=(args.expected_size,args.expected_size))
    if args.external:
        other=audit_directory(args.external,expected_size=(args.expected_size,args.expected_size))
        result['external']={k:v for k,v in other.items() if k!='sha256_hashes'}
        result['hash_overlap_count_with_external']=len(cross_dataset_overlap(result,other))
    result.pop('sha256_hashes',None)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    dump({'images':result['image_count'],'classes':result['class_counts'],
          'corrupt':len(result['corrupt_files']),
          'duplicate_groups':len(result['duplicate_files_by_sha256']),
          'report':args.output})


def extract_command(args):
    import zipfile
    expected={'NCT-CRC-HE-100K.zip':'6fd702d11df6292bc054397ae038a464',
              'NCT-CRC-HE-100K-NONORM.zip':'035777cf327776a71a05c95da6d6325f',
              'CRC-VAL-HE-7K.zip':'2fd1651b4f94ebd818ebf90ad2b6ce06'}
    zip_path=Path(args.archive)
    if zip_path.name not in expected:
        raise ValueError('Only the three verified original Zenodo archive names are supported')
    md5=hashlib.md5()  # nosec: used only to verify Zenodo data archive integrity, not security
    with zip_path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            md5.update(block)
    if md5.hexdigest()!=expected[zip_path.name]:
        raise ValueError('Checksum mismatch; see official Zenodo record for correct file')
    dest=Path(args.dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            path=(dest/member.filename).resolve()
            if not path.is_relative_to(dest):
                raise ValueError(f'Unsafe zip member: {member.filename}')
        z.extractall(dest)
    dump({'extracted_to':str(dest),'archive_md5_verified':True,
          'advice':'Identify the directory whose immediate subfolders are the nine tissue classes.'})


def cache_command(args):
    from onco_agent.evaluation.cache import inference_cache
    output=inference_cache(args.model,args.output,manifest=args.manifest,subset=args.subset,
        external_root=args.external_root,batch_size=args.batch_size,workers=args.workers,
        device=args.device,max_items=args.max_items)
    dump({'cache':str(output),'meta':str(output.with_suffix('.json'))})


def infer_command(args):
    import torch
    from onco_agent.models.factory import load_checkpoint
    from onco_agent.data.datasets import image_transform
    from onco_agent.tools.image_quality import quality_indicators
    from onco_agent.agents.policy import Policy
    from onco_agent.agents.learned import LearnedPolicy
    from onco_agent.calibration.temperature import apply_temperature
    from onco_agent.models.training import choose_device
    device=choose_device(args.device)
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    model, meta=load_checkpoint(args.fast_model,device)
    strong, meta2=load_checkpoint(args.strong_model,device)
    if meta['classes']!=meta2['classes']:
        raise ValueError('Class ordering differs between two model checkpoints')
    with Image.open(args.image) as im:
        rgb=im.convert('RGB')
        quality=quality_indicators(np.asarray(rgb.resize((64,64))))
        a=image_transform(meta['size'])(rgb).unsqueeze(0).to(device)
        b=image_transform(meta2['size'])(rgb).unsqueeze(0).to(device)
    with torch.inference_mode():
        fast=apply_temperature(model(a).cpu().numpy(),cfg['calibration']['fast_temperature'])[0]
        def strong_fn():
            with torch.inference_mode():
                return apply_temperature(strong(b).cpu().numpy(),cfg['calibration']['strong_temperature'])[0]
        if args.policy=='learned':
            policy=LearnedPolicy.load(cfg['learned_policy_file'])
            trace=policy.run(fast,strong_fn,quality={
                'brightness':quality['mean_brightness'],
                'edge_variance':quality['edge_variance'],
                'saturation':quality['saturation_fraction']})
        else:
            trace=Policy(args.policy,cfg['fast_threshold'],cfg['review_threshold']).run(
                fast,strong_fn, quality_flag=quality['edge_variance']<cfg['blur_edge_threshold'])
    dump({'research_only':True,'not_for_clinical_use':True,
          'classes':meta['classes'],'fast_class_probabilities':dict(zip(meta['classes'],map(float,fast))),
          'quality_indicators':quality,'actions':trace.actions,
          'abstained':trace.abstained,'strong_calls':trace.strong_tool_calls,
          'predicted_tissue_class':None if trace.predicted_class is None else meta['classes'][trace.predicted_class],
          'warning':'Non-diagnostic tissue-label prediction from a single patch; calibration is not a clinical risk score.'})


def demo_data(args):
    # Image colours encode labels to make a reproducible PIPELINE test, not research evidence.
    rng=np.random.default_rng(args.seed); root=Path(args.output)
    from onco_agent.data.datasets import CLASS_CODES
    if root.exists() and any(root.rglob('*.png')):
        raise FileExistsError(f'Images already exist under {root}; refusing overwrite')
    for i,code in enumerate(CLASS_CODES):
        folder=root/code;folder.mkdir(parents=True,exist_ok=True)
        for k in range(args.count):
            base=np.zeros((args.size,args.size,3),dtype=np.uint8)
            base[:,:,i%3]=min(240,70+(i//3)*75)
            noise=rng.integers(0,28,base.shape,dtype=np.uint8)
            Image.fromarray(np.clip(base.astype(int)+noise,0,255).astype(np.uint8)).save(folder/f'{code}_{k:03}.png')
    dump({'root':str(root),'synthetic_only':True,'samples':9*args.count,
          'warning':'Synthetic colour patterns are ONLY for software smoke tests, not medical performance.'})


def main(argv=None):
    p=argparse.ArgumentParser(prog='onco-agent',description='ONCO-AGENT research-only CLI. NOT FOR CLINICAL USE.')
    sub=p.add_subparsers(dest='cmd',required=True)
    x=sub.add_parser('audit');x.add_argument('--root',required=True);x.add_argument('--external')
    x.add_argument('--expected-size',type=int,default=224);x.add_argument('--output',default='results/audit.json');x.set_defaults(fn=audit_command)
    x=sub.add_parser('extract');x.add_argument('--archive',required=True);x.add_argument('--dest',default='data/raw');x.set_defaults(fn=extract_command)
    x=sub.add_parser('split');x.add_argument('--root',required=True);x.add_argument('--output',default='results/splits.json')
    x.add_argument('--groups-csv');x.add_argument('--seed',type=int,default=42);x.add_argument('--policy-fraction',type=float,default=.15)
    x.add_argument('--val-fraction',type=float,default=.15);x.add_argument('--image-size',type=int,default=224);x.set_defaults(fn=split_command)
    x=sub.add_parser('train');x.add_argument('--manifest',required=True);x.add_argument('--output',required=True)
    x.add_argument('--architecture',choices=['resnet18','efficientnet_b0','tiny_cnn'],default='resnet18')
    x.add_argument('--epochs',type=int,default=8);x.add_argument('--batch-size',type=int,default=8)
    x.add_argument('--lr',type=float,default=.0001);x.add_argument('--size',type=int,default=224)
    x.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');x.add_argument('--pretrained',action='store_true')
    x.add_argument('--workers',type=int,default=0);x.add_argument('--seed',type=int,default=42)
    x.add_argument('--accumulation',type=int,default=1);x.add_argument('--patience',type=int,default=4)
    x.add_argument('--amp',action='store_true',help='Opt in to CUDA mixed precision (default disabled for stability)')
    x.add_argument('--grad-clip',type=float,default=1.0)
    x.add_argument('--freeze-bn',action='store_true',help='Keep pretrained BatchNorm running statistics fixed during training (recommended for EfficientNet with microbatch=2)')
    from onco_agent.models.training import train
    x.set_defaults(fn=lambda a: train(a.manifest,a.output,architecture=a.architecture,epochs=a.epochs,
        batch_size=a.batch_size,lr=a.lr,size=a.size,device=a.device,pretrained=a.pretrained,
        workers=a.workers,seed=a.seed,accumulation=a.accumulation,patience=a.patience,
        amp=a.amp,grad_clip=a.grad_clip,freeze_bn=a.freeze_bn))
    x=sub.add_parser('cache');x.add_argument('--model',required=True);x.add_argument('--output',required=True)
    x.add_argument('--manifest');x.add_argument('--subset',choices=['policy','val'])
    x.add_argument('--external-root');x.add_argument('--batch-size',type=int,default=16)
    x.add_argument('--workers',type=int,default=0);x.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    x.add_argument('--max-items',type=int);x.set_defaults(fn=cache_command)
    x=sub.add_parser('calibrate');x.add_argument('--fast',required=True);x.add_argument('--strong',required=True)
    x.add_argument('--output',required=True)
    from onco_agent.evaluation.study import calibrate_val,fit_from_cache,freeze_config,evaluate
    x.set_defaults(fn=lambda a:dump(calibrate_val(a.fast,a.strong,a.output)))
    x=sub.add_parser('fit-policy');x.add_argument('--fast',required=True);x.add_argument('--strong',required=True)
    x.add_argument('--calibration',required=True);x.add_argument('--output',required=True)
    x.add_argument('--strong-cost',type=float,default=.1);x.add_argument('--review-cost',type=float,default=.25)
    x.add_argument('--wrong-penalty',type=float,default=2.)
    x.set_defaults(fn=lambda a:dump(fit_from_cache(a.fast,a.strong,a.calibration,a.output,
                    strong_cost=a.strong_cost,review_cost=a.review_cost,wrong_penalty=a.wrong_penalty)))
    x=sub.add_parser('freeze');x.add_argument('--calibration',required=True);x.add_argument('--learned',required=True)
    x.add_argument('--output',required=True);x.add_argument('--fast-threshold',type=float,default=.8)
    x.add_argument('--review-threshold',type=float,default=.65);x.add_argument('--blur-edge-threshold',type=float,default=70.)
    x.add_argument('--strong-cost',type=float,default=.1);x.add_argument('--review-cost',type=float,default=.25)
    x.add_argument('--wrong-penalty',type=float,default=2.)
    x.set_defaults(fn=lambda a:dump(freeze_config(a.calibration,a.learned,a.output,a.fast_threshold,
                        a.review_threshold,a.blur_edge_threshold,a.strong_cost,a.review_cost,a.wrong_penalty)))
    x=sub.add_parser('evaluate');x.add_argument('--fast',required=True);x.add_argument('--strong',required=True)
    x.add_argument('--config',required=True);x.add_argument('--output',required=True)
    x.add_argument('--split',choices=['val','external'],required=True)
    x.set_defaults(fn=lambda a:dump(evaluate(a.fast,a.strong,a.config,a.output,split=a.split)))
    x=sub.add_parser('infer');x.add_argument('--image',required=True);x.add_argument('--fast-model',required=True)
    x.add_argument('--strong-model',required=True);x.add_argument('--config',required=True)
    x.add_argument('--policy',choices=['rule','fast','fixed_abstain','learned'],default='rule')
    x.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');x.set_defaults(fn=infer_command)
    x=sub.add_parser('plot');x.add_argument('--metrics',required=True);x.add_argument('--output',required=True)
    from onco_agent.reporting.figures import plot_policy_summary
    x.set_defaults(fn=lambda a:plot_policy_summary(a.metrics,a.output))
    x=sub.add_parser('stress');x.add_argument('--model',required=True)
    x.add_argument('--external-root',required=True);x.add_argument('--config',required=True)
    x.add_argument('--kind',choices=['blur','dark','bright','color'],required=True)
    x.add_argument('--output',required=True);x.add_argument('--max-items',type=int)
    x.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    from onco_agent.evaluation.stress import stress_test
    x.set_defaults(fn=lambda a:dump(stress_test(a.model,a.external_root,a.config,a.kind,a.output,
                                                 device=a.device,max_items=a.max_items)))
    x=sub.add_parser('synthetic-data');x.add_argument('--output',default='data/synthetic_smoke')
    x.add_argument('--count',type=int,default=16);x.add_argument('--size',type=int,default=64)
    x.add_argument('--seed',type=int,default=42);x.set_defaults(fn=demo_data)
    args=p.parse_args(argv)
    args.fn(args)

if __name__=='__main__':
    main()
