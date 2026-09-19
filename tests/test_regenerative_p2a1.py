import copy
import pickle
import cloudpickle
import numpy as np
import pytest
import torch
from research.regenerative_control.continuing import Config
from research.regenerative_control.continuing_p11 import PostDeliveryUAV
from research.regenerative_control.qualify import actor_for
from research.regenerative_control.census_p2a1 import compact,enumerated,ForbiddenRNG,physical

@pytest.fixture(scope='module')
def actor():
    torch.set_num_threads(1)
    e=PostDeliveryUAV(Config(obstacles=0)); a=actor_for(e);e.close();return a

def test_enumeration_never_reads_rng_and_uses_explicit_task(actor):
    e=PostDeliveryUAV(Config(obstacles=0))
    a=enumerated(compact(e),100)
    with pytest.raises(RuntimeError): a.task_rng.integers(3)
    r=a.advance_home(actor)
    assert r['reward']==1 and a.events[-1]['delivered_task']['index']==0
    assert 'task_rng' not in r['start_state'] and 'task_rng' not in r['next_macro_state']
    assert a.phase=='decision' and a.task is None
    a.close();e.close()

def test_log_compaction_and_rng_changes_do_not_change_branch(actor):
    e=PostDeliveryUAV(Config(obstacles=0))
    b=enumerated(compact(e),100);b.advance_home(actor)
    b.task_rng=np.random.default_rng(1001)
    raw=cloudpickle.dumps(b)
    c=pickle.loads(compact(b));c.task_rng=np.random.default_rng(9001)
    x=enumerated(raw,300);y=enumerated(compact(c),300)
    assert x.execute('C',actor)==y.execute('C',actor)
    assert physical(x)==physical(y)
    for z in [e,b,c,x,y]:z.close()

def test_energy_intervention_is_separate_and_original_unchanged():
    e=PostDeliveryUAV(Config(obstacles=0))
    e.base.agent.energy=3
    raw=compact(e)
    normal=enumerated(raw,600);counterfactual=enumerated(raw,600,True)
    assert normal.energy==3 and e.energy==3 and counterfactual.energy==10000
    assert normal.time==counterfactual.time==e.time
    np.testing.assert_equal(normal.base.agent.pos,counterfactual.base.agent.pos)
    for z in [e,normal,counterfactual]:z.close()

def test_snapshot_roundtrip_keeps_physical_key_and_rng():
    e=PostDeliveryUAV(Config(obstacles=0))
    c=pickle.loads(compact(e))
    assert physical(e)==physical(c)
    assert e.task_rng.bit_generator.state==c.task_rng.bit_generator.state
    e.close();c.close()


def test_diagnostic_capacity_keeps_paid_service_time_positive(actor):
    e=PostDeliveryUAV(Config(obstacles=0))
    b=enumerated(compact(e),100,True)
    c=b.advance_home(actor)
    r=b.execute('R',actor)
    assert c['success'] and r['success']
    assert r['elapsed_time'] >= b.config.recharge_overhead
    assert b.energy == b.config.capacity == 10000
    assert e.energy == e.config.capacity == 60
    e.close();b.close()
