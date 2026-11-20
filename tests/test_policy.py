import numpy as np
import pytest
from onco_agent.agents.policy import Policy
from onco_agent.agents.learned import fit_policy,LearnedPolicy


def test_tool_called_only_on_uncertainty():
    called=[]
    def strong():
        called.append(1)
        return np.array([.6,.4])
    result=Policy('rule').run([.9,.1],strong)
    assert result.strong_tool_calls==0
    assert not called
    assert result.predicted_class==0
    result=Policy('rule').run([.5,.5],strong)
    assert result.strong_tool_calls==1
    assert len(called)==1


def test_disagreement_escalates():
    result=Policy('rule').run([.55,.45],lambda:np.array([.05,.95]))
    assert result.abstained and result.predicted_class is None
    assert 'review' in result.actions


def test_budget_and_invalid_input():
    result=Policy('rule',max_strong_calls=0).run([.51,.49],lambda:pytest.fail('not allowed'))
    assert result.strong_tool_calls==0
    with pytest.raises(ValueError):
        Policy('rule').run([3,7])
    with pytest.raises(ValueError):
        Policy('rule',max_strong_calls=4)


def test_learned_policy_never_uses_truth_at_decision_time(tmp_path):
    rng=np.random.default_rng(21)
    f=rng.dirichlet([2,2,2],size=50);s=rng.dirichlet([2,2,2],size=50)
    labels=np.arange(50)%3
    q=[{'brightness':.5,'edge_variance':60.,'saturation':.01} for _ in labels]
    learned,costs=fit_policy(f,s,labels,q)
    path=tmp_path/'policy.json';learned.save(path,costs)
    loaded=LearnedPolicy.load(path)
    assert loaded.choose(f[0],q[0]) in ('ACCEPT_FAST','CALL_STRONG','REVIEW')
    # Inference does not accept labels and is stable if labels elsewhere change.
    assert loaded.choose(f[0],q[0])==learned.choose(f[0],q[0])
    assert not costs.get('reinforcement_learning',False)
