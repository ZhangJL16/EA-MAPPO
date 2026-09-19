import copy
import pickle

import numpy as np
import pytest
import torch

from research.regenerative_control.continuing import Config
from research.regenerative_control.continuing_p11 import PostDeliveryUAV
from research.regenerative_control.collect_p1 import dump


class ZeroActor:
    def __call__(self, obs, deterministic=True):
        return torch.zeros((1,3))


def complete_legs(monkeypatch, env):
    """Isolate task lifecycle ordering; real SAC traces test physical execution."""
    def leg(goal, actor, name):
        env.base.simulation_time += 1
        return dict(goal_reached=True)
    monkeypatch.setattr(env, '_navigate', leg)


def test_initial_home_is_task_free_and_has_no_cr_action():
    e=PostDeliveryUAV(Config(obstacles=0))
    assert e.phase=='home' and e.task is None and e.task_draw_count==0
    assert not hasattr(e,'abandoned')
    for a in ['C','R']:
        with pytest.raises(ValueError): e.execute(a,ZeroActor())
    assert e.task_draw_count==0 and e.time==0
    e.close()


def test_delivery_enters_task_free_decision_without_extra_rng_draw(monkeypatch):
    e=PostDeliveryUAV(Config(obstacles=0)); complete_legs(monkeypatch,e)
    rng=copy.deepcopy(e.task_rng); expected=int(rng.integers(3))
    r=e.advance_home(ZeroActor())
    assert r['action']=='forced_task' and r['reward']==1
    assert e.phase=='decision' and e.task is None and e.task_draw_count==1
    assert e.task_rng.bit_generator.state==rng.bit_generator.state
    assert e.events[-1]['delivered_task']['index']==expected
    assert e.events[-1]['task'] is None and e.events[-1]['phase']=='decision'
    e.close()


def test_only_committed_c_draws_one_task_in_d_state(monkeypatch):
    e=PostDeliveryUAV(Config(obstacles=0)); complete_legs(monkeypatch,e)
    e.advance_home(ZeroActor())
    rng=copy.deepcopy(e.task_rng); expected=int(rng.integers(3))
    with pytest.raises(ValueError): e.execute('invalid',ZeroActor())
    assert e.task_draw_count==1
    r=e.execute('C',ZeroActor())
    assert r['start_state']['task'] is None
    assert [x['event'] for x in e.events][-3:]==['decision_C','task_draw','delivery_complete']
    assert e.events[-1]['delivered_task']['index']==expected
    assert e.task_draw_count==2 and e.task_rng.bit_generator.state==rng.bit_generator.state
    assert e.phase=='decision' and e.task is None
    e.close()


def test_r_draws_nothing_and_first_home_task_is_after_recharge(monkeypatch):
    e=PostDeliveryUAV(Config(obstacles=0)); complete_legs(monkeypatch,e)
    e.advance_home(ZeroActor())
    rng=copy.deepcopy(e.task_rng.bit_generator.state)
    e.execute('R',ZeroActor())
    assert e.task_rng.bit_generator.state==rng and e.task_draw_count==1
    assert e.phase=='home' and e.task is None and not hasattr(e,'abandoned')
    assert e.events[-1]['event']=='recharge_complete' and e.events[-1]['task'] is None
    np.testing.assert_array_equal(e.base.agent.pos,e.base.charger_position)
    np.testing.assert_array_equal(e.base.agent.vel,[0,0,0])
    assert e.energy==60 and e.base.reset_calls==1
    with pytest.raises(ValueError): e.execute('R',ZeroActor())
    event_count=len(e.events)
    e.advance_home(ZeroActor())
    assert e.events[event_count]['event']=='forced_first_task'
    assert e.events[event_count+1]['event']=='task_draw'
    assert e.task_draw_count==2 and e.task is None
    e.close()


def test_pending_task_cannot_be_rejected_or_replaced():
    e=PostDeliveryUAV(Config(obstacles=0))
    e._draw_task()
    e.phase='decision'  # invalid externally injected pending task must fail closed
    for a in ['C','R']:
        with pytest.raises(ValueError): e.execute(a,ZeroActor())
    assert e.task_draw_count==1
    e.close()


def test_home_snapshot_resume_preserves_next_draw_and_real_cutoff(tmp_path):
    e=PostDeliveryUAV(Config(obstacles=0,cutoff=.2))
    p=tmp_path/'home.pkl'; dump(p,e)
    with p.open('rb') as f: restored=pickle.load(f)
    a=e.advance_home(ZeroActor()); b=restored.advance_home(ZeroActor())
    assert a==b and e.events==restored.events and e.audit()==restored.audit()
    assert e.truncated and e.task_draw_count==1 and e.base.reset_calls==1
    e.close(); restored.close()
