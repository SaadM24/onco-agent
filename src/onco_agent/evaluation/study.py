"""Frozen-study config and policy replay. No test labels are exposed to action selection."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from onco_agent.agents.policy import Policy
from onco_agent.agents.learned import LearnedPolicy
from onco_agent.calibration.temperature import fit_temperature, apply_temperature
from onco_agent.evaluation.cache import paired_caches
from onco_agent.evaluation.metrics import classification_metrics


def quality_dict(cache, i):
    return {'brightness': float(cache['brightness'][i]),
            'edge_variance': float(cache['edge_variance'][i]),
            'saturation': float(cache['saturation'][i])}


def calibrate_val(fast_cache, strong_cache, output):
    fast, strong = paired_caches(fast_cache, strong_cache)
    meta = json.loads(Path(fast_cache).with_suffix('.json').read_text(encoding='utf-8'))
    meta2 = json.loads(Path(strong_cache).with_suffix('.json').read_text(encoding='utf-8'))
    if meta['source_kind'] != 'val' or meta2['source_kind'] != 'val':
        raise ValueError('Calibration must ONLY use internal val caches')
    setting = {'fast_temperature': fit_temperature(fast['logits'], fast['labels']),
               'strong_temperature': fit_temperature(strong['logits'], strong['labels']),
               'source': 'internal_val_only', 'external_used_to_fit': False}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setting, indent=2), encoding='utf-8')
    return setting


def get_probs(cache, temperature):
    return apply_temperature(cache['logits'], temperature)


def fit_from_cache(fast_cache, strong_cache, calibration_path, output,
                   strong_cost=.1, review_cost=.25, wrong_penalty=2.):
    from onco_agent.agents.learned import fit_policy
    a,b = paired_caches(fast_cache,strong_cache)
    m1 = json.loads(Path(fast_cache).with_suffix('.json').read_text(encoding='utf-8'))
    m2 = json.loads(Path(strong_cache).with_suffix('.json').read_text(encoding='utf-8'))
    if m1['source_kind'] != 'policy' or m2['source_kind'] != 'policy':
        raise ValueError('Learned policy must ONLY be fit on internal policy partition')
    cal = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    learner, costs = fit_policy(get_probs(a,cal['fast_temperature']),
                                get_probs(b,cal['strong_temperature']), a['labels'],
                                [quality_dict(a,i) for i in range(len(a['labels']))],
                                strong_cost=strong_cost, review_cost=review_cost,
                                wrong_penalty=wrong_penalty)
    learner.save(output,costs)
    return costs


def freeze_config(calibration_path, learned_path, output, fast_threshold=.80,
                  review_threshold=.65, blur_edge_threshold=70., strong_cost=.1,
                  review_cost=.25, wrong_penalty=2.):
    if not (0 <= review_threshold <= fast_threshold <= 1):
        raise ValueError('Expect 0 <= review_threshold <= fast_threshold <= 1')
    # User picks and freezes thresholds BEFORE external evaluation. These are defaults,
    # NOT claimed optimized thresholds; calibration fit uses val labels.
    cal = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    data = {'calibration': cal, 'learned_policy_file': str(Path(learned_path).resolve()),
            'fast_threshold': fast_threshold, 'review_threshold': review_threshold,
            'blur_edge_threshold': blur_edge_threshold, 'strong_cost': strong_cost,
            'review_cost': review_cost, 'wrong_penalty': wrong_penalty,
            'protocol_frozen_before_external': True,
            'limitation': 'Internal validation was used for calibration; report its metrics as tuning diagnostics, not independent estimates.'}
    output.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return data


def _policy_route(name, f, s, q, cfg, learned):
    if name == 'strong':
        return int(s.argmax()), False, ('strong_predict',), 1
    if name == 'learned':
        t = learned.run(f, lambda: s, quality=q)
    else:
        model = Policy(mode=name, fast_threshold=cfg['fast_threshold'],
                       review_threshold=cfg['review_threshold'])
        t = model.run(f, lambda: s, quality_flag=q['edge_variance'] < cfg['blur_edge_threshold'])
    return t.predicted_class, t.abstained, t.actions, t.strong_tool_calls


def evaluate(fast_cache, strong_cache, frozen_config, output_dir, split='external'):
    a, b = paired_caches(fast_cache, strong_cache)
    m1 = json.loads(Path(fast_cache).with_suffix('.json').read_text(encoding='utf-8'))
    m2 = json.loads(Path(strong_cache).with_suffix('.json').read_text(encoding='utf-8'))
    if m1['source_kind'] != split or m2['source_kind'] != split or split not in {'val','external'}:
        raise ValueError('Evaluation split mismatched; expected val or external')
    if split == 'external' and (m1.get('incomplete_subset', False) or m2.get('incomplete_subset', False)):
        raise ValueError('Incomplete external caches cannot be reported as final external evaluation')
    cfg = json.loads(Path(frozen_config).read_text(encoding='utf-8'))
    if not cfg['protocol_frozen_before_external']:
        raise ValueError('Unfrozen protocol')
    cal = cfg['calibration']
    f = get_probs(a, cal['fast_temperature'])
    s = get_probs(b, cal['strong_temperature'])
    y = np.asarray(a['labels'], dtype=int)
    metrics = {'split': split,
               'note': 'No patient-level inference from image patches. Validation metrics are selection-biased.' if split == 'val' else
               'External evaluation: do not tune after viewing these results.',
               'fast_classifier': classification_metrics(f, y),
               'strong_classifier': classification_metrics(s, y),
               'fast_uncalibrated': classification_metrics(softmax(a['logits'],axis=1), y),
               'strong_uncalibrated': classification_metrics(softmax(b['logits'],axis=1), y)}
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    trace_path = out / f'{split}_traces.csv'
    with trace_path.open('w', newline='', encoding='utf-8') as file:
        fields = ['sample_id','policy','ground_truth','predicted_class','accepted','correct_if_accepted','actions','strong_calls','confidence']
        wr = csv.DictWriter(file, fieldnames=fields); wr.writeheader()
        for name in ('fast','strong','fixed_abstain','rule','learned'):
            learner = LearnedPolicy.load(cfg['learned_policy_file']) if name == 'learned' else None
            accepted, correct, calls, costs = 0, 0, 0, 0.
            matrix = np.zeros((len(y), f.shape[1]), dtype=float)
            for i in range(len(y)):
                q=quality_dict(a,i)
                pred, abstain, actions, nstrong = _policy_route(name,f[i],s[i],q,cfg,learner)
                accepted += int(not abstain); calls += nstrong
                if not abstain:
                    correct += int(pred == y[i])
                # Baseline always runs fast; strong-only requires only strong.
                costs += nstrong * cfg['strong_cost'] + abstain * cfg['review_cost']
                chosen = s[i] if name == 'strong' or nstrong else f[i]
                matrix[i] = chosen
                wr.writerow({'sample_id': str(a['ids'][i]), 'policy': name,
                             'ground_truth': int(y[i]),
                             'predicted_class': '' if abstain else int(pred),
                             'accepted': int(not abstain),
                             'correct_if_accepted': '' if abstain else int(pred == y[i]),
                             'actions': '|'.join(actions), 'strong_calls': nstrong,
                             'confidence': round(float(chosen.max()), 6)})
            coverage = accepted / len(y)
            metrics[name] = {'samples': len(y), 'coverage': coverage,
                             'abstention_rate': 1-coverage,
                             'accepted_accuracy': correct/accepted if accepted else None,
                             'selective_error': 1-correct/accepted if accepted else None,
                             'strong_calls_per_image': calls/len(y),
                             'modeled_extra_cost_per_image': costs/len(y),
                             'note': 'Modeled costs are unitless parameters, not real wall-clock time.'}
    (out / f'{split}_metrics.json').write_text(json.dumps(metrics, indent=2),encoding='utf-8')
    return metrics
