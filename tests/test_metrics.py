import numpy as np
import pytest
from onco_agent.calibration.temperature import fit_temperature,apply_temperature
from onco_agent.evaluation.metrics import classification_metrics,selective_metrics
from onco_agent.evaluation.robustness import perturb
from PIL import Image


def test_calibration_positive_and_normalized():
    z=np.array([[2,0],[0,3],[3,0],[0,1]],dtype=float)
    y=[0,1,0,1]
    t=fit_temperature(z,y)
    p=apply_temperature(z,t)
    assert t>0 and np.allclose(p.sum(1),1)
    assert classification_metrics(p,y)['accuracy']==1
    with pytest.raises(ValueError):apply_temperature(z,0)


def test_selective_metric_no_accepted():
    p=np.array([[.8,.2],[.6,.4]])
    z=selective_metrics(p,[0,1],[True,True])
    assert z['coverage']==0 and z['accepted_accuracy'] is None


def test_robustness_modifies_image():
    im=Image.new('RGB',(32,32),(125,125,125))
    assert perturb(im,'dark').getpixel((0,0))[0]<125
    with pytest.raises(ValueError):perturb(im,'unknown')
