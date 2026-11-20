"""Full simulated cache -> calibration -> fit -> freeze -> replay, no hospital data."""
import json
from pathlib import Path
import numpy as np
import pytest
from onco_agent.evaluation.study import calibrate_val,fit_from_cache,freeze_config,evaluate


def cache(path,role,logits,labels):
    n=len(labels)
    np.savez_compressed(path,ids=np.array([f'i_{i}' for i in range(n)]),labels=labels,
            logits=logits,brightness=np.ones(n)*.6,edge_variance=np.ones(n)*100.,saturation=np.zeros(n))
    path.with_suffix('.json').write_text(json.dumps({'source_kind':role,'classes':['a','b']}))


def test_complete_research_replay(tmp_path):
    rng=np.random.default_rng(23)
    a=rng.normal(size=(44,2));b=rng.normal(size=(44,2));y=np.arange(44)%2
    paths={}
    for role in ('policy','val','external'):
        fa=tmp_path/f'{role}_fast.npz';fb=tmp_path/f'{role}_strong.npz'
        cache(fa,role,a,y);cache(fb,role,b,y)
        paths[role]=(fa,fb)
    cal=tmp_path/'calibration.json'
    calibrate_val(*paths['val'],cal)
    with pytest.raises(ValueError):calibrate_val(*paths['external'],cal)
    learner=tmp_path/'learner.json'
    fit_from_cache(*paths['policy'],cal,learner)
    with pytest.raises(ValueError):fit_from_cache(*paths['external'],cal,learner)
    frozen=tmp_path/'frozen.json'
    freeze_config(cal,learner,frozen)
    summary=evaluate(*paths['external'],frozen,tmp_path/'study','external')
    assert summary['fast']['coverage']==1
    assert summary['learned']['strong_calls_per_image']<=1
    assert (tmp_path/'study/external_traces.csv').exists()
    assert (tmp_path/'study/external_metrics.json').exists()
