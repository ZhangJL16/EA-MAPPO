import copy
import pickle

import numpy as np
import pytest
import torch

from research.regenerative_control.continuing import Config, ContinuingUAV
from research.regenerative_control.collect_p1 import dump, summarize


class ZeroActor:
    def __call__(self, obs, deterministic=True):
        assert obs.shape == (1,2056)
        return torch.zeros((1,3))


def test_task_goal_switch_keeps_physical_state_and_contact_memory():
    e=ContinuingUAV(Config(obstacles=0))
    b=e.base
    b.agent.vel[:]=[1,2,0]
    b.agent.collided=True
    before=(b.agent.pos.copy(),b.agent.vel.copy(),e.energy,e.time,b.current_step)
    e._set_goal(np.array(e.task['dropoff']))
    np.testing.assert_array_equal(b.agent.pos,before[0])
    np.testing.assert_array_equal(b.agent.vel,before[1])
    assert (e.energy,e.time,b.current_step)==before[2:]
    assert b.agent.collided and b.reset_calls==1
    e.close()


def test_delivery_never_resets_or_refills_and_draws_next_task():
    e=ContinuingUAV(Config(obstacles=0))
    # Already inside both service goals isolates high-level delivery semantics.
    e.task['pickup']=e.base.agent.pos.tolist()
    e.task['dropoff']=e.base.agent.pos.tolist()
    e.base.agent.energy=17
    e.base.simulation_time=123
    e.base.agent.vel[:]=[1,2,0]
    r=e.execute('C',ZeroActor())
    assert r['reward']==1 and e.completed==1 and e.phase=='decision'
    assert e.energy==17 and e.time==123 and e.base.reset_calls==1
    np.testing.assert_array_equal(e.base.agent.vel,[1,2,0])
    assert [x['event'] for x in e.events][-2:]==['delivery_complete','task_available']
    e.close()


def test_regeneration_is_paid_canonical_and_rng_not_reseeded():
    e=ContinuingUAV(Config(obstacles=0))
    e.base.agent.energy=20
    e.base.cumulative_virtual_energy=40
    e.base.agent.pos += [2,0,0]
    e.base.agent.prev_pos=e.base.agent.pos.copy()
    e.base.agent.vel[:]=[2,1,0]
    e.base.agent.collided=True
    rng=copy.deepcopy(e.task_rng)
    expected=int(rng.integers(3))
    r=e.execute('R',ZeroActor())
    assert r['success'] and e.base.reset_calls==1
    assert e.time>52 and e.energy==60
    np.testing.assert_array_equal(e.base.agent.pos,e.base.charger_position)
    np.testing.assert_array_equal(e.base.agent.vel,[0,0,0])
    assert not e.base.agent.collided
    assert e.task['index']==expected
    assert e.task_rng.bit_generator.state==rng.bit_generator.state
    event=[x for x in e.events if x['event']=='recharge_complete'][0]
    assert event['task'] is None and event['phase']=='regenerated'
    assert abs(e.audit()['energy_balance_residual'])<1e-8
    assert e.cycles[0]['initial_delayed_cycle']
    e.close()


def test_cutoff_censors_charge_without_free_full_battery():
    e=ContinuingUAV(Config(obstacles=0,cutoff=12))
    e.base.agent.energy=20
    r=e.execute('R',ZeroActor())
    assert e.truncated and not e.failed and e.energy==22 and e.time==12
    assert not e.cycles and r['terminal_reason']=='evaluation_cutoff'
    assert e.base.reset_calls==1
    e.close()


def test_short_cutoff_remainder_does_not_erase_completed_regeneration():
    e=ContinuingUAV(Config(obstacles=0,cutoff=10.2))
    r=e.execute('R',ZeroActor())
    assert r['success'] and len(e.cycles)==1 and e.time<10.2
    assert e.energy==60 and abs(e.audit()['energy_balance_residual'])<1e-8
    e.close()


def test_depletion_is_failure_without_reset_and_midtask_decision_rejected():
    e=ContinuingUAV(Config(obstacles=0))
    e.base.agent.energy=1e-12
    e.execute('C',ZeroActor())
    assert e.failed and e.failure_reason=='energy_depletion'
    assert e.base.reset_calls==1 and e.completed==0
    with pytest.raises(ValueError): e.execute('R',ZeroActor())
    e.close()


def test_atomic_snapshot_resume_matches_uninterrupted_service(tmp_path):
    e=ContinuingUAV(Config(obstacles=0))
    e.base.agent.energy=17
    e.base.cumulative_virtual_energy=43
    path=tmp_path/'snapshot.pkl'
    dump(path,dict(env=e))
    with path.open('rb') as f: restored=pickle.load(f)['env']
    a=e.execute('R',ZeroActor())
    b=restored.execute('R',ZeroActor())
    assert a==b and e.events==restored.events and e.audit()==restored.audit()
    e.close(); restored.close()


def test_failed_expenditure_is_not_a_completion_quantile():
    rows=[dict(state_group='near_light',option='task',success=False,
        audit={'failed':True},collision=False,energy_used=.1,elapsed_time=2,
        legs=[{'elapsed_time':2}])]
    r=summarize(rows)[0]
    assert r['q95_unconditional_completion_energy'] is None
    assert r['q95_energy_success'] is None and r['failures']==1


def test_locked_contact_penalties_and_repair_in_new_plant():
    e=ContinuingUAV(Config(obstacles=0))
    b=e.base
    def hit():
        b.agent.pos[0]=.51; b.agent.prev_pos=b.agent.pos.copy()
        b.agent.vel[:]=[-20,0,0]
        _,_,done,_,info=b.step(np.array([-1,0,0],np.float32))
        assert not done and info['boundary_contact']
        return info['reward_components']['boundary_penalty_component']
    assert hit()==-1.2
    assert hit()==-.42
    b.agent.pos[0]=100; b.agent.prev_pos=b.agent.pos.copy(); b.agent.vel[:]=0
    b.step(np.zeros(3,np.float32))
    assert hit()==-1.2
    e.close()
