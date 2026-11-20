"""Generate calibration/reliability and confusion plot from saved metrics.
Only executed research results should go into figures.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_policy_summary(metrics_path, dest):
    m=json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    names=[i for i in ('fast','strong','fixed_abstain','rule','learned') if i in m]
    coverage=[m[n]['coverage'] for n in names]
    errors=[np.nan if m[n]['selective_error'] is None else m[n]['selective_error'] for n in names]
    fig,ax=plt.subplots(figsize=(8,5))
    for i,n in enumerate(names):
        ax.scatter(coverage[i],errors[i],s=60)
        ax.annotate(n,(coverage[i],errors[i]),xytext=(5,5),textcoords='offset points')
    ax.set_xlim(0,1.05);ax.set_ylim(bottom=0)
    ax.set_xlabel('Coverage (fraction accepted)');ax.set_ylabel('Selective error on accepted predictions')
    ax.set_title('Decision policy comparison — research only')
    fig.tight_layout();Path(dest).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=160);plt.close(fig)
