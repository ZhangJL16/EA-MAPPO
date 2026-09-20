import json
import math
import pickle
from pathlib import Path
import unittest
from single_life_rl.core.model import VersionSet
from single_life_rl.core.experiment import Experiment
from single_life_rl.core.refinement import observational_partition, common_answers
from single_life_rl.core.information import design
from single_life_rl.algorithms.safe_refine import run_toy
from single_life_rl.envs.uav_mission_library import MissionModels, run_uav

CONFIG=json.loads((Path(__file__).parents[1]/'configs/protocol_v1.json').read_text())


class CoreTests(unittest.TestCase):
    def test_refinement_hand_computed(self):
        models=[(0,0),(0,1),(1,0),(1,1)]
        blocks=observational_partition(models,[0],lambda m,e:m[e])
        self.assertEqual(blocks,[[(0,0),(0,1)],[(1,0),(1,1)]])
        self.assertEqual(observational_partition(blocks[0],[1],lambda m,e:m[e]),[[(0,0)],[(0,1)]])

    def test_anytime_likelihood_ratio_martingale(self):
        for k in (.02,.04,.08,.16):
            e=Experiment('p',0,k)
            for true in (0,1):
                expected=sum(e.observation_likelihood(true,y)*e.observation_likelihood(1-true,y)/e.observation_likelihood(true,y) for y in (0,1))
                self.assertAlmostEqual(expected,1.)
        self.assertLessEqual(sum(.05*2**(-(i+1)) for i in range(34)),.05)
        for m in (4,8,16,32):
            self.assertAlmostEqual(.05/2+.05/4+m*(.05/(4*m)),.05)

    def test_no_budget_reset_and_pickle_resume(self):
        e=Experiment('p',0,.08);v=VersionSet(2)
        for y in [1,0,1]*20:v.update(e,y,.025)
        clone=pickle.loads(pickle.dumps(v))
        for y in [0,1]*20:
            v.update(e,y,.025);clone.update(e,y,.025)
        self.assertEqual(v,clone)

    def test_unlock_only_after_confidence(self):
        e=Experiment('p',0,.08);child=Experiment('child',1,.08,prerequisites=((0,1),))
        v=VersionSet(2);self.assertFalse(child.safe(v))
        while 0 not in v.known:v.update(e,1,.025)
        self.assertTrue(child.safe(v));self.assertGreaterEqual(v.log_odds[0],math.log(40))

    def test_independent_bit_design(self):
        es=[Experiment('a',0,.04,2),Experiment('b',1,.08,3)]
        weights,gamma=design(es)
        den=sum(weights[e.name]*e.duration for e in es)
        self.assertAlmostEqual(weights['a']*es[0].information()/den,gamma)
        self.assertAlmostEqual(weights['b']*es[1].information()/den,gamma)

    def test_zero_information_stays_locked(self):
        j=dict(job_id='test',suite='binary',method='SafeRefine',truth=[1],seed=1,horizon=100,kappa=0.)
        r=run_toy(j,CONFIG)
        self.assertFalse(r['catastrophe']);self.assertIsNone(r['certification_time']);self.assertEqual(r['regret'],80.)
        self.assertEqual(r['safe_probe_count'],0)

    def test_nuisance_does_not_change_decision_learning(self):
        common=dict(job_id='test',suite='nuisance',method='SafeRefine',seed=1,horizon=50000,kappa=.08)
        a=run_toy(dict(common,truth=[0,1],m=0),CONFIG)
        b=run_toy(dict(common,truth=[0,1]+[0]*32,m=32),CONFIG)
        self.assertEqual(a['certification_time'],b['certification_time'])
        self.assertEqual(a['exploration_time'],b['exploration_time'])

    def test_static_cannot_unlock_second_bit(self):
        j=dict(job_id='test',suite='recursive',method='StaticSafeID',truth=[0,1],seed=2,horizon=30000,kappas=[.08,.08])
        r=run_toy(j,CONFIG)
        self.assertIsNone(r['certification_time']);self.assertFalse(r['catastrophe'])
        self.assertEqual(r['unlocked_experiments'],0)

    def test_safe_refine_full_id_identical_without_nuisance(self):
        j=dict(job_id='test',suite='nuisance',truth=[1,0],seed=3,horizon=50000,kappa=.08,m=0)
        a=run_toy(dict(j,method='SafeRefine'),CONFIG);b=run_toy(dict(j,method='FullModelID'),CONFIG)
        for field in ('catastrophe','certification_time','exploration_time','reward'):
            self.assertEqual(a[field],b[field])

    def test_uav_clock_filters_without_gaussian_underflow(self):
        lib=dict(missions=[dict(mission_id=0,energy=10.,duration=20.,success=True)],battery=20.,recharge_rate=1.)
        m=MissionModels(lib,CONFIG['uav']['theta']);arm=m.arms[0]
        clock=m.cycle_duration(2,arm)
        self.assertEqual([i for i in range(5) if m.cycle_duration(i,arm)==clock],[2])
        j=dict(job_id='test',method='SafeRefine',truth_index=2,seed=1,horizon=100)
        r=run_uav(j,CONFIG,lib)
        self.assertEqual(r['certification_time'],0.);self.assertFalse(r['catastrophe'])


if __name__=='__main__':unittest.main()
