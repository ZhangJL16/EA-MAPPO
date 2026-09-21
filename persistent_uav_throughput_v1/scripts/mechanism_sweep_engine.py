"""Branch-local interventions; original runtime and running IQ v1 stay immutable."""
from dataclasses import asdict,replace
from collections import Counter
import math,pickle
import real_state_branching_atlas as atlas
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.baselines import Action
from persistent_uav.streams import stream_hash
census=atlas.census
BIG=1e9
CONDITIONS=('C1','C2','C3','C4','C5','C6','C7','E','D1','D2','K8','K12')

def set_energy(nav,value):
    if hasattr(nav,'base'):nav.base.agent.energy=float(value)
    else:nav.energy=float(value)

def ledger(env):
    return float(env.nav.base.cumulative_virtual_energy) if hasattr(env.nav,'base') else float(getattr(env,'_energy_meter',0.))

class SweepEnvironment(PersistentUAVThroughput):
    def instant_fill(self):
        set_energy(self.nav,self.config.capacity);self.completed_recharges+=1;self.mode='IDLE'
        self._event('charge_complete',battery=self.nav.energy,intervention='instant_charge')
    def _arrived_at_goal(self):
        super()._arrived_at_goal()
        if self.mode=='CHARGING' and self._instant and not self._oracle_context:self.instant_fill()
    @property
    def can_recharge(self):return False if self._no_energy else super().can_recharge
    def step(self,action=None):
        before=self.nav.energy
        if self._instant and not self._oracle_context and self.mode=='CHARGING' and not self.done:
            assert action is None
            self.instant_fill();self._arrive_until(self.time);self._assert_invariants()
            result=(self.observe(),0,self.done)
        else:result=super().step(action)
        self._energy_meter+=max(0.,before-self.nav.energy)
        if self._no_energy:assert self.nav.energy>BIG/2,'energy-free sentinel bound invalid'
        return result
    def _assert_invariants(self):
        if not self._erased:return super()._assert_invariants()
        assert len(self.queue)<=self.config.queue_capacity
        assert -1e-8<=self.nav.energy<=self.config.capacity+1e-8
        assert math.isfinite(self.time) and self.time<=self.config.cutoff+1e-7
        assert self.arrivals==self.accepted+self.overflow
        assert self.initial_accepted+self.accepted==self.completed+len(self.queue)+int(self.current_task is not None)+self._erased


def apply(env,condition):
    original_hash=env.workload_hash
    if condition in ('C1','C6','K8','K12'):
        cap=max(5,len(env._tasks)) if condition in ('C1','C6') else int(condition[1:])
        env.config=replace(env.config,queue_capacity=cap);env.queue_time.extend([0.]*(cap+1-len(env.queue_time)))
    if condition!='C1':
        env.__class__=SweepEnvironment;env._instant=condition in ('C2','C6');env._no_energy=condition in ('C4','C7');env._oracle_context=False;env._erased=0;env._energy_meter=0.
    if condition in ('C3','C7'):
        env._tasks=env._tasks[:env.cursor];env.workload_hash=stream_hash(env._tasks)
    if condition in ('C4','C7'):
        env.config=replace(env.config,capacity=BIG);env.nav.capacity=BIG
        if hasattr(env.nav,'base'):env.nav.base.configure_calibrated_battery(BIG,reserve_fraction=0,source='mechanism_sweep_no_energy')
        else:set_energy(env.nav,BIG)
    return original_hash


def start(root,first,condition,erase=None):
    c=atlas.new_continuation(root,first);env=c['env'];original=apply(env,condition)
    c.update(condition=condition,erase=erase,original_stream_hash=original,base_steps=env.nav.policy_steps,start_energy_ledger=ledger(env),start_queue_time=list(env.queue_time),start_recharges=env.recharge_count,root_queue=[asdict(t) for t in env.queue],regeneration_started=False)
    return c

def choose(safe,condition):
    if condition=='D1':return min(safe,key=lambda r:(r['task_energy'],r['spec']['task_id']))
    if condition=='D2':return min(safe,key=lambda r:(-r['actual_reserve'],r['spec']['task_id']))
    return min(safe,key=lambda r:(r['prediction']['predicted_task_time'],r['spec']['task_id']))


def finish(c,reason):
    r=atlas.continuation_result(c,reason);e=c['env']
    r.update(condition=c['condition'],original_stream_sha256=c['original_stream_hash'],active_stream_sha256=e.workload_hash,root_queue=c['root_queue'],root_queue_capacity=e.config.queue_capacity,physical_energy_consumed=ledger(e)-c['start_energy_ledger'],recharge_requests=e.recharge_count-c['start_recharges'],queue_occupancy_integral=sum(i*(v-c['start_queue_time'][i]) for i,v in enumerate(e.queue_time)),erased_tasks=getattr(e,'_erased',0))
    return r

def advance(c,model):
    cond=c['condition']
    if cond=='C1':
        r=atlas.advance_continuation(c,model)
        return finish(c,r['outcome']) if r is not None else None
    env=c['env'];noenergy=cond in ('C4','C7')
    if env.done or env.time>=c['deadline']-1e-7:
        return finish(c,env.failure or ('window_end' if env.time>=c['deadline']-1e-7 else 'original_horizon_censored'))
    if c['stage']=='initial':
        action=None if c['initial_started'] else Action('serve',c['first']['spec']['task_id'],'atlas_first_safe_task')
        c['initial_started']=True;before=len(env.events);census.set_rng(c['rng']);atlas.window_step(env,action,c['deadline']);c['rng']=census.rng_state()
        completion=next((e for e in env.events[before:] if e['event']=='task_completed'),None)
        if completion is not None:
            actual=dict(completion);expected=dict(c['first']['task_completion'])
            if noenergy:actual.pop('battery');expected.pop('battery')
            census.assert_equal(actual,expected,'sweep first-task physical path')
            c['initial_completion']=completion;c['stage']='forced_regeneration' if cond=='C5' else 'physical'
            if cond=='E':
                task=next(t for t in env.queue if t.id==c['erase']);env._queue_integrate(env.time);env.queue.remove(task);env._erased+=1
                env._event('counterpart_erased',task_id=task.id,reward=0);env._assert_invariants()
        return None
    if c['stage']=='forced_regeneration':
        if not c['regeneration_started']:
            if atlas.station(env) and env.nav.energy>=env.config.capacity-1e-9:
                c['stage']='physical';return None
            c['pending_regen_count']=env.completed_recharges+1;c['regeneration_started']=True
            action=Action('recharge',reason='sweep_post_first_forced_regeneration');c['forced_returns']+=1;c['forced_return_reasons']['post_first_forced_regeneration']+=1
        else:action=None
        census.set_rng(c['rng']);atlas.window_step(env,action,c['deadline']);c['rng']=census.rng_state()
        if env.completed_recharges>=c['pending_regen_count']:c['stage']='physical'
        return None
    if c['stage']=='oracle':
        if c['oracle'] is None:
            c['oracle']=census.new_branch(atlas.clone_root(env,c['rng']),c['oracle_specs'][c['oracle_index']]);c['oracle']['env']._oracle_context=True
        b=c['oracle'];before=b['env'].nav.policy_steps
        if noenergy:
            action=None if b['started'] else Action('serve',b['spec']['task_id'],'privileged_census_candidate');b['started']=True
            # new_branch restores the root RNG; the worker checkpoints the
            # ongoing global RNG, exactly as the archived task-return oracle.
            old=len(b['env'].events);b['env'].step(action)
            b['oracle_policy_steps']+=b['env'].nav.policy_steps-before
            completion=next((e for e in b['env'].events[old:] if e['event']=='task_completed'),None)
            result=None
            if completion:b['task_completion']=completion;result=census.branch_result(b,'success',True);result['reached_charger']=False;result['safety_predicate']='task_only'
            elif b['env'].failure:result=census.branch_result(b,b['env'].failure,False)
            elif b['env'].done:result=census.branch_result(b,'horizon_censored',None)
            assert b['oracle_policy_steps']<=env.config.option_step_limit
        else:result=atlas.advance_one(b)
        c['oracle_physics']+=b['env'].nav.policy_steps-before
        if result is None:return None
        if noenergy:
            result['safety_predicate']='task_only'
            result['reached_charger']=False
        result=atlas.enriched(result,env.observe(),model)
        if noenergy:
            # A return was not tested: absent measurements must not look like
            # zero-cost successful returns or enormous certified reserves.
            result.update(return_time=None,return_energy=None,actual_reserve=None)
        c['oracle_results'].append(result);b['env'].nav.close();c['oracle']=None;c['oracle_index']+=1
        if c['oracle_index']<len(c['oracle_specs']):return None
        safe=[r for r in c['oracle_results'] if r['spec']['kind']=='task_return' and r['safe'] is True]
        if safe:
            chosen=choose(safe,cond);action=Action('serve',chosen['spec']['task_id'],'atlas_oracle_safe_frozen_sjf' if cond not in ('D1','D2') else 'sweep_oracle_safe_'+cond)
        elif not noenergy and env.can_recharge and next(r for r in c['oracle_results'] if r['spec']['kind']=='direct_return')['safe'] is True:
            reason='empty_queue_return' if not env.queue else 'no_safe_task_return';action=Action('recharge',reason=reason);c['forced_returns']+=1;c['forced_return_reasons'][reason]+=1
        else:action=None
        c['oracle_decisions'].append(dict(time=env.time,observation=env.observe(),action=asdict(action) if action else None,branches=c['oracle_results']));c['oracle_results']=[]
        if action is None:return finish(c,'no_safe_continuation')
        c['pending_action']=action;c['stage']='physical';return None
    if env.decision_required and 'pending_action' not in c:
        if not env.queue and (noenergy or atlas.station(env)):c['pending_action']=Action('idle',reason='sweep_no_energy_wait' if noenergy else 'atlas_empty_queue_at_station_wait')
        else:
            c['oracle_specs']=atlas.specs(env) if env.queue else [dict(kind='direct_return',task_id=None)]
            if noenergy:c['oracle_specs']=[s for s in c['oracle_specs'] if s['kind']=='task_return']
            c['oracle_index']=0;c['oracle_results']=[];c['stage']='oracle';return None
    action=c.pop('pending_action',None);census.set_rng(c['rng']);atlas.window_step(env,action,c['deadline']);c['rng']=census.rng_state()
    return None
