import hashlib
import itertools
import json
import pytest
from research.regenerative_control.stopping_dp import solve
from research.regenerative_control.stopping_p2a2 import robust_continue,envelope,read_envelope,expand


def toy(times):
    nodes={};roots=[]
    for i,t in enumerate(times):
        key=f'd{i}';children=[]
        for j in range(3):
            child=f'{key}_{j}'
            nodes[child]=dict(R_safe=True,tau_R=1.,safe_C=False,children=[])
            children.append(dict(id=child,time=t,probability=1/3))
        nodes[key]=dict(R_safe=True,tau_R=1.,safe_C=True,children=children)
        roots.append(dict(id=key,time=1.,probability=1/3))
    return nodes,roots


def test_ratio_dp_matches_exhaustive_policies_and_early_return():
    times=[100.,1.,1.];nodes,roots=toy(times)
    result=solve(nodes,roots,'d0')
    rates=[]
    for policy in itertools.product([False,True],repeat=3):
        n=sum(2 if a else 1 for a in policy)/3
        t=sum(2+times[i] if a else 2 for i,a in enumerate(policy))/3
        rates.append(n/t)
    assert result['rho_star']==pytest.approx(max(rates))
    assert result['rho_star']==pytest.approx(5/8)
    assert result['actions']['d0']=='R' and result['candidate']['safe_C']
    assert result['VB_over_oracle']<.98 and result['decision']=='GAP_BELOW_98'
    assert result['bracket_excess'][0]>=0>=result['bracket_excess'][1]


def test_viability_policy_kills_when_it_is_optimal():
    nodes,roots=toy([1.,1.,1.]);r=solve(nodes,roots)
    assert r['VB_over_oracle']==pytest.approx(1)
    assert r['decision']=='KILL'
    assert not r['safe_return_optimal_nodes']


def test_ties_are_not_strict_safe_but_return_optimal():
    nodes,roots=toy([2.,2.,2.]);r=solve(nodes,roots)
    assert r['rho_star']==pytest.approx(.5)
    assert not r['safe_return_optimal_nodes']


def test_incomplete_safe_tree_cannot_be_solved():
    nodes,roots=toy([1.,1.,1.]);del nodes['d0_0']
    with pytest.raises(ValueError,match='incomplete'):solve(nodes,roots)


def test_safe_set_requires_all_tasks_and_actual_return_success():
    def case(task=True,ret=True):
        return dict(task=dict(success=task),return_after_task=dict(row=dict(success=ret)))
    assert robust_continue([case(),case(),case()])
    assert not robust_continue([case(),case(),case(ret=False)])
    assert not robust_continue([case(),case(),case(task=False)])
    assert not robust_continue([case(),case()])


def test_checkpoint_reuse_and_corruption_detection(tmp_path):
    (tmp_path/'nodes').mkdir();raw=b'opaque physical snapshot'
    (tmp_path/'state.pkl').write_bytes(raw)
    d=dict(id='node',snapshot='state.pkl',snapshot_sha256=hashlib.sha256(raw).hexdigest())
    row=dict(descriptor=d,children=[])
    path=tmp_path/'nodes/node.json';envelope(path,row)
    assert expand((str(tmp_path),d,{}))==row  # cached path never invokes actor
    (tmp_path/'state.pkl').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='snapshot'):expand((str(tmp_path),d,{}))
    e=json.loads(path.read_text());e['row']['children']=['corrupt'];path.write_text(json.dumps(e))
    with pytest.raises(RuntimeError,match='corrupt receipt'):read_envelope(path)


def test_structural_report_excludes_candidate_duplicate_and_checks_gap_identity():
    from research.regenerative_control.report_p2a2 import summarize
    nodes,roots=toy([100.,1.,1.]);nodes['candidate']=nodes['d0'].copy()
    r=solve(nodes,roots,'candidate');r['complete_tree']=True;r['counts']={}
    r['node_metadata']={k:dict(origin='candidate' if k=='candidate' else 'cycle',
                              sequence=[k],state=dict(test_state=k)) for k in nodes}
    s=summarize(r)
    assert s['safe_return_optimal_cycle_prefixes']==1
    assert s['distinct_physical_early_return_states']==1
    assert s['optimal_probability_of_early_return_per_cycle']==pytest.approx(1/3)
    assert s['largest_single_prefix_loss_fraction']==pytest.approx(1)
    assert abs(s['independent_loss_identity_residual'])<1e-10
