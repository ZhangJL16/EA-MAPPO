"""Intervention contracts on the existing engineering fixture, no research runs."""
import pickle,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mechanism_sweep_engine as sweep
from test_real_state_atlas import AtlasTests
from test_core import task
from persistent_uav.baselines import Action

class SweepTests(unittest.TestCase):
    def setUp(self): self.helper=AtlasTests()
    def run_condition(self,condition,tasks=None,erase=None):
        e=self.helper.env(tasks=tasks);first=self.helper.one(e,0)
        with patch.object(sweep.atlas,'WINDOW',20.):
            c=sweep.start(sweep.atlas.clone_root(e,sweep.census.rng_state()),first,condition,erase)
            for _ in range(4000):
                r=sweep.advance(c,self.helper.model())
                if r is not None:return c,r
        self.fail('did not finish')
    def test_noenergy_task_only_and_wait_in_place(self):
        c,r=self.run_condition('C7',tasks=[task(0,10),task(1,20)])
        self.assertEqual(r['N_W'],2);self.assertEqual(r['recharge_requests'],0)
        self.assertEqual(c['env'].nav.position.tolist(),[20.,0.,0.])
        queries=[b for d in r['oracle_decisions'] for b in d['branches']]
        self.assertTrue(queries);self.assertTrue(all(b['spec']['kind']=='task_return' and b['safety_predicate']=='task_only' and not b['reached_charger'] for b in queries))
        self.assertFalse(any(e['event'] in ('charger_arrival','charge_complete') for e in r['events']))
    def test_noarrivals_preserves_root_and_history(self):
        e=self.helper.env();before=e.observe();cursor=e.cursor;events=list(e.events)
        sweep.apply(e,'C3')
        self.assertEqual(e.observe(),before);self.assertEqual(e.events,events)
        self.assertEqual(len(e._tasks),cursor)
    def test_erasure_no_reward_and_conservation(self):
        c,r=self.run_condition('E',tasks=[task(0,10),task(1,20)],erase=1)
        self.assertEqual(r['N_W'],1);self.assertEqual(r['erased_tasks'],1);c['env']._assert_invariants()
    def test_forced_regeneration_once(self):
        c,r=self.run_condition('C5',tasks=[task(0,10),task(1,20)])
        self.assertEqual(r['forced_return_reasons']['post_first_forced_regeneration'],1)
    def test_instant_fill_main_but_oracle_stops_before_fill(self):
        for oracle in [False,True]:
            e=self.helper.env();sweep.apply(e,'C2');e._oracle_context=oracle
            e.step(Action('serve',0))
            while e.mode=='SERVING':e.step()
            e.step(Action('recharge'))
            while e.mode=='RETURNING':e.step()
            self.assertEqual(e.mode,'CHARGING' if oracle else 'IDLE')
            self.assertEqual(e.completed_recharges,0 if oracle else 1)
    def test_noenergy_nested_resume(self):
        e=self.helper.env();first=self.helper.one(e,0)
        with patch.object(sweep.atlas,'WINDOW',20.):
            c=sweep.start(sweep.atlas.clone_root(e,sweep.census.rng_state()),first,'C4')
            for _ in range(100):
                sweep.advance(c,self.helper.model())
                if c['oracle'] is not None:break
            self.assertIsNotNone(c['oracle']);state=pickle.dumps((c,sweep.census.rng_state()))
            def finish(c):
                for _ in range(4000):
                    r=sweep.advance(c,self.helper.model())
                    if r is not None:return r
                self.fail('resume timeout')
            a=finish(c);d,rng=pickle.loads(state);sweep.census.set_rng(rng);b=finish(d)
            self.assertEqual(sweep.census.normalized(a),sweep.census.normalized(b))
if __name__=='__main__':unittest.main()

class CollectionTests(unittest.TestCase):
    def test_prefix_counts_endpoint_and_unknown(self):
        import mechanism_intervention_sweep as runner
        r=dict(start_time=10.,physical_end_time=20.,failure_time=None,failure=None,events=[dict(time=10.,event='decision',action=dict(kind='serve',task_id=1)),dict(time=15.,event='task_completed',task_id=1)])
        a=runner.prefix(r,5.,[dict(id=1)]);b=runner.prefix(r,11.,[dict(id=1)])
        self.assertEqual(a['N'],1);self.assertEqual(a['flight_time'],5.)
        self.assertIsNone(b['N']);self.assertFalse(b['known'])
    def test_known_absorbing_failure(self):
        import mechanism_intervention_sweep as runner
        r=dict(start_time=0.,physical_end_time=5.,failure_time=5.,failure='navigation_failure',events=[])
        self.assertEqual(runner.prefix(r,10.,[])['N'],0)
        self.assertFalse(runner.prefix(r,10.,[])['survived'])
    def test_all_ranking_ties_use_id(self):
        candidates=[dict(task_energy=2.,actual_reserve=3.,prediction=dict(predicted_task_time=4.),spec=dict(task_id=i)) for i in (2,1)]
        for condition in ('D1','D2','C2'):self.assertEqual(sweep.choose(candidates,condition)['spec']['task_id'],1)

class AdditionalContracts(unittest.TestCase):
    def test_c1_engine_matches_existing_intervention(self):
        import infinite_queue_intervention as iq
        h=AtlasTests();e=h.env();first=h.one(e,0);root=sweep.atlas.clone_root(e,sweep.census.rng_state())
        with patch.object(sweep.atlas,'WINDOW',20.):
            a=sweep.start(root,first,'C1');b=sweep.atlas.new_continuation(root,first);iq.unbounded(b['env'])
            def finish(c,step):
                for _ in range(4000):
                    r=step(c,h.model())
                    if r is not None:return r
                self.fail('parity timeout')
            left=finish(a,sweep.advance);right=finish(b,sweep.atlas.advance_continuation)
            self.assertEqual({k:sweep.census.normalized(left[k]) for k in right},sweep.census.normalized(right))
    def test_capacity_doses_and_combination(self):
        h=AtlasTests()
        for cond,cap,instant in [('K8',8,False),('K12',12,False),('C6',5,True)]:
            e=h.env();sweep.apply(e,cond)
            self.assertEqual(e.config.queue_capacity,cap);self.assertEqual(len(e.queue_time),cap+1);self.assertEqual(e._instant,instant)
