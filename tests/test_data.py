import json
import numpy as np
import pytest
from PIL import Image
from onco_agent.data.datasets import load_folder,make_internal_splits,write_splits,ManifestDataset
from onco_agent.data.audit import audit_directory,cross_dataset_overlap


def test_non_leaky_internal_splits(tmp_path):
    for k in ('ADI','TUM','STR'):
        p=tmp_path/'original'/k;p.mkdir(parents=True)
        for i in range(24):
            Image.fromarray(np.full((32,32,3),i*9%255,dtype=np.uint8)).save(p/f'{i:02}.png')
    ds=load_folder(tmp_path/'original',size=32)
    splits=make_internal_splits(ds,seed=17)
    assert set(splits)=={'train','policy','val'}
    assert len(set(sum(splits.values(),[])))==len(ds)
    filename=tmp_path/'splits.json'
    assert len(write_splits(ds,splits,filename,17))==64
    m=json.loads(filename.read_text())
    assert not m['group_independent']
    assert not m['external_dataset_used_for_selection']
    assert len(ManifestDataset(filename,'policy',size=32))==len(splits['policy'])
    with pytest.raises(ValueError):
        ManifestDataset(filename,'external')


def test_audit_duplicate_and_damage(tmp_path):
    path=tmp_path/'ADI';path.mkdir()
    im=Image.new('RGB',(16,16),(250,3,2))
    im.save(path/'a.png');im.save(path/'b.png')
    (path/'c.png').write_bytes(b'not-a-png')
    a=audit_directory(tmp_path,expected_size=(224,224))
    assert a['image_count']==3
    assert len(a['duplicate_files_by_sha256'])==1
    assert len(a['corrupt_files'])==1
    assert len(a['unexpected_shapes'])==2
    assert len(cross_dataset_overlap(a,a))==1


def test_group_isolation(tmp_path):
    for k in ('ADI','TUM'):
        p=tmp_path/k;p.mkdir()
        for i in range(40):
            Image.new('RGB',(32,32),(i,i,30)).save(p/f'{i}.png')
    ds=load_folder(tmp_path,size=32)
    groups=[str(i//2)+'-'+ds.classes[label] for i,label in enumerate(ds.targets)]
    splits=make_internal_splits(ds,groups=groups,seed=33)
    assert all(set(groups[i] for i in splits[a]).isdisjoint(set(groups[i] for i in splits[b]))
               for a,b in [('train','policy'),('train','val'),('policy','val')])
