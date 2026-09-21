"""Engineering checks only; fixtures are not research environments."""
import copy
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import real_state_branching_atlas as atlas
from test_core import TestNavigator,task
from persistent_uav.config import Config
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.estimates import EstimateModel
from persistent_uav.storage import snapshot,restore

class Navigator(TestNavigator):
    def close(self): pass

class AtlasTests(unittest.TestCase):
    def env(self,capacity=20.,tasks=None):
        tasks=tasks if tasks is not None else [task(0,10),task(1,20),task(2,30,3.)]
        return PersistentUAVThroughput(Config(capacity,2.,.2,30.,initial_tasks=sum(t.arrival==0 for t in tasks)),Navigator(capacity),tasks)
    def model(self): return EstimateModel([0.,.1,0.],[0.,.1,0.])
    def one(self,env,task_id):
        b=atlas.census.new_branch(atlas.clone_root(env,atlas.census.rng_state()),dict(kind='task_return',task_id=task_id))
        for _ in range(2000):
            r=atlas.advance_one(b)
            if r is not None:return atlas.enriched(r,env.observe(),self.model())
        self.fail('branch did not resolve')
    def finish(self,c):
        for _ in range(3000):
            r=atlas.advance_continuation(c,self.model())
            if r is not None:return r
        self.fail('continuation did not resolve')
    def test_hash_sampling_ignores_failure_and_order(self):
        obs=self.env().observe();obs['remaining_time']=2000.
        runs={f'B5/r{s}':dict(summary=dict(regime=0,seed=s,stream_sha256='x',failure=True),events=[dict(event='decision',time=0.,observation=obs,action={})]) for s in range(8)}
        roots,cells=atlas.select_roots(runs)
        changed=copy.deepcopy(dict(reversed(list(runs.items()))))
        for r in changed.values():r['summary']['failure']=False
        self.assertEqual(roots,atlas.select_roots(changed)[0]);self.assertEqual(len(roots),4)
        self.assertEqual(sum(c['selected'] for c in cells),4)
    def test_eligibility_and_short_cell(self):
        obs=self.env().observe();obs['remaining_time']=atlas.WINDOW-.05
        runs={'B5/r':dict(summary=dict(regime=0,seed=1,stream_sha256='x'),events=[dict(event='decision',time=0,observation=obs,action={})])}
        self.assertEqual(atlas.select_roots(runs)[0],[])
        obs['remaining_time']=atlas.WINDOW
        self.assertEqual(len(atlas.select_roots(runs)[0]),1)
    def test_docked_return_no_illegal_recharge(self):
        env=self.env();before=pickle.dumps(env)
        b=atlas.census.new_branch(atlas.clone_root(env,atlas.census.rng_state()),dict(kind='direct_return',task_id=None))
        r=atlas.advance_one(b)
        self.assertTrue(r['safe']);self.assertEqual(r['duration'],0.);self.assertEqual(r['events'],[])
        self.assertEqual(pickle.dumps(env),before)
    def test_actual_energy_accounting(self):
        r=self.one(self.env(),0)
        self.assertTrue(r['safe']);self.assertAlmostEqual(r['task_time'],1.)
        self.assertAlmostEqual(r['task_energy'],1.);self.assertAlmostEqual(r['return_energy'],1.)
        self.assertAlmostEqual(r['actual_reserve'],18.)
    def test_stationary_window_keeps_original_stream(self):
        env=self.env(tasks=[task(0,10,20.)]);original=env.config
        atlas.window_step(env,None,5.)
        self.assertEqual(env.time,5.);self.assertTrue(env.done);self.assertIs(env.config,original)
        self.assertEqual(env.cursor,0);self.assertEqual(len(env._tasks),1)
    def test_first_task_counts_at_endpoint(self):
        env=self.env();row=self.one(env,0)
        with patch.object(atlas,'WINDOW',1.):
            r=self.finish(atlas.new_continuation(atlas.clone_root(env,atlas.census.rng_state()),row))
        self.assertEqual(r['N_W'],1);self.assertTrue(r['F_W'])
    def test_short_window_before_first_completion(self):
        env=self.env();row=self.one(env,1)
        with patch.object(atlas,'WINDOW',.5):
            r=self.finish(atlas.new_continuation(atlas.clone_root(env,atlas.census.rng_state()),row))
        self.assertEqual(r['N_W'],0);self.assertTrue(r['F_W'])
    def test_no_safe_is_unknown_not_death(self):
        env=self.env(capacity=3.,tasks=[task(0,10),task(1,1000)]);row=self.one(env,0)
        with patch.object(atlas,'WINDOW',20.):
            r=self.finish(atlas.new_continuation(atlas.clone_root(env,atlas.census.rng_state()),row))
        self.assertTrue(r['no_safe_continuation']);self.assertIsNone(r['N_W']);self.assertIsNone(r['F_W'])
        self.assertFalse(r['depletion']);self.assertEqual(r['N_observed'],1)
    def test_nested_disk_resume_and_parent_isolation(self):
        env=self.env();row=self.one(env,0);root=atlas.clone_root(env,atlas.census.rng_state());before=pickle.dumps(root)
        with patch.object(atlas,'WINDOW',8.):
            c=atlas.new_continuation(root,row)
            for _ in range(100):
                atlas.advance_continuation(c,self.model())
                if c['oracle'] is not None:break
            self.assertIsNotNone(c['oracle'])
            with tempfile.TemporaryDirectory() as path:
                snapshot(path,dict(c=c,rng=atlas.census.rng_state()));saved=restore(path)
            left=self.finish(c);atlas.census.set_rng(saved['rng']);right=self.finish(saved['c'])
        self.assertEqual(atlas.census.normalized(left),atlas.census.normalized(right))
        self.assertEqual(pickle.dumps(root),before);self.assertEqual(c['env']._tasks,root['env']._tasks)
    def test_unresolved_not_zero_regret(self):
        env=self.env();a=self.one(env,0);b=self.one(env,1)
        r=atlas.analyze_root(dict(id='r',policy='B5',regime=0),dict(one_step=[a,b],continuations=[dict(task_id=0,N_W=3),dict(task_id=1,N_W=None)]))
        self.assertFalse(r['fully_resolved']);self.assertIsNone(r['delta_N'])
        self.assertTrue(all(h['regret'] is None for h in r['heuristics'].values()))

if __name__=='__main__':unittest.main()
