"""Engineering only: reuse existing test plant, not a research environment."""
import sys,pickle,unittest
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import infinite_queue_intervention as intervention
from test_core import TestNavigator,task
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.config import Config
from persistent_uav.baselines import Action

class Tests(unittest.TestCase):
    def env(self):
        tasks=[task(i,10*(i+1)) for i in range(5)]+[task(i,100,.05*(i-4)) for i in range(5,10)]
        return PersistentUAVThroughput(Config(100,1,.2,30,initial_tasks=5),TestNavigator(100),tasks)
    def test_only_capacity_and_accounting_change(self):
        e=self.env();old=asdict(e.config);before=e.observe();bins=list(e.queue_time);hash0=e.workload_hash
        cap=intervention.unbounded(e);new=asdict(e.config)
        self.assertEqual({k for k in old if old[k]!=new[k]},{'queue_capacity'})
        self.assertEqual(e.observe(),before);self.assertEqual(e.queue_time[:6],bins);self.assertEqual(e.workload_hash,hash0)
        self.assertEqual(cap,len(e._tasks));self.assertEqual(len(e.queue_time),cap+1)
    def test_accept_every_future_task_without_changing_first_physics(self):
        finite=self.env();large=pickle.loads(pickle.dumps(finite));intervention.unbounded(large)
        for e in [finite,large]:
            e.step(Action('serve',0))
            while e.mode=='SERVING':e.step()
        self.assertGreater(finite.overflow,0);self.assertEqual(large.overflow,0);self.assertGreater(len(large.queue),5)
        self.assertEqual(finite.time,large.time);self.assertEqual(finite.nav.energy,large.nav.energy)
        self.assertEqual(finite.nav.position.tolist(),large.nav.position.tolist())
        self.assertEqual(finite.workload_hash,large.workload_hash)
    def test_no_resurrection_after_historical_drop(self):
        e=self.env();e.step(Action('serve',0))
        while e.mode=='SERVING':e.step()
        ids=[t.id for t in e.queue];cursor=e.cursor;overflow=e.overflow;events=list(e.events)
        intervention.unbounded(e)
        self.assertEqual(ids,[t.id for t in e.queue]);self.assertEqual(e.cursor,cursor);self.assertEqual(e.overflow,overflow);self.assertEqual(e.events,events)
    def test_clone_isolation_and_idempotence(self):
        e=self.env();before=pickle.dumps(e);clone=pickle.loads(before);intervention.unbounded(clone)
        self.assertEqual(pickle.dumps(e),before);cap=clone.config.queue_capacity;bins=list(clone.queue_time)
        intervention.unbounded(clone);self.assertEqual(clone.config.queue_capacity,cap);self.assertEqual(clone.queue_time,bins)
if __name__=='__main__':unittest.main()
