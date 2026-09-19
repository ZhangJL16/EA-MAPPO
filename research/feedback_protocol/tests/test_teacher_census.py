import copy
import importlib.util
from fractions import Fraction as F
from pathlib import Path
from dataclasses import replace
import json
import tempfile
import unittest
from unittest.mock import patch
from fpl.problem import PublicProblem,Operation,UtilitySpec
from fpl.protocols import enumerate_protocols
from fpl.belief import PlannerState,outcomes
from fpl.teacher_data import advance,state_key
from fpl.evaluation import save_new

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('census',ROOT/'scripts/census_teacher_v2.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
PLAN=json.loads((ROOT/'configs/teacher_v2_census_v1.json').read_text())


def fixture():
    z=UtilitySpec()
    return PublicProblem('tiny','tiny','train',('q','x'),'q',('a','b'),
        ((F(1,2),F(3,4)),(F(1,2),F(1,4))),(F(1,2),F(1,2)),
        (Operation('a','q','q',1,1,('a',),z),
         Operation('a_slow','q','q',2,1,('a',),z),
         Operation('joint','q','q',2,2,('a','b'),z),
         Operation('task','q','q',1,1,(),UtilitySpec(by_hypothesis=(F(1),F(0)))),
         Operation('empty','q','q',1,1,(),z)),3,4,2,objective='cumulative_task_utility')


class CensusTests(unittest.TestCase):
    def test_full_history_dfs_matches_dedup_closure(self):
        p=fixture();result=mod.closure(p,PLAN)
        self.assertEqual(result['status'],'complete')
        found={};root=PlannerState(p.prior,p.budget,p.capacity)
        def dfs(state,depth):
            found.setdefault(state_key(state),set()).add(depth)
            if depth==2:return
            for route in enumerate_protocols(p,state.remaining):
                if mod.measuring_only(p,route):
                    for feedback,_,_ in outcomes(p,state.posterior,route.channels):
                        dfs(advance(p,state,route,feedback),depth+1)
        dfs(root,0)
        self.assertEqual(found,{r['id']:set(r['depths']) for r in result['records']})
        self.assertTrue(any(set(r['depths'])=={1,2} for r in result['records']))
        self.assertTrue(any(e['operations']==['joint'] for e in result['edges']))
        self.assertFalse(any('task' in e['operations'] or 'empty' in e['operations'] for e in result['edges']))
        self.assertEqual(mod.independent_closure_check(p,PLAN,result)['status'],'complete_checked')
        broken=copy.deepcopy(result);broken['edges'].pop()
        with self.assertRaises(AssertionError):mod.independent_closure_check(p,PLAN,broken)

    def test_caps_and_reward_observation_boundary(self):
        p=fixture();plan=copy.deepcopy(PLAN);plan['closure_limits']['max_states']=1
        result=mod.closure(p,plan)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(mod.independent_closure_check(p,plan,result)['status'],'incomplete_not_certified')
        legacy=replace(p,operations=tuple(replace(o,utility=None) for o in p.operations))
        self.assertFalse(any(mod.measuring_only(legacy,r) for r in enumerate_protocols(legacy,4)))

    def test_resume_refuses_interrupted_work(self):
        p=fixture();raw=mod.from_json # keep import exercised
        from fpl_v2.generator import as_json
        old=dict(problem=as_json(p),metadata={'root_seed':1},records=[])
        with tempfile.TemporaryDirectory() as d,patch.object(mod,'load_pilot',return_value=[old]):
            out=Path(d)/'run'
            mod.run('mock',PLAN,out,max_new=0)
            save_new(out/'tiny.started.json',dict(interrupted=True))
            with self.assertRaises(RuntimeError):mod.run('mock',PLAN,out,resume=True)
        with tempfile.TemporaryDirectory() as d,patch.object(mod,'load_pilot',return_value=[old]):
            out=Path(d)/'run';mod.run('mock',PLAN,out,max_new=1)
            original=(out/'tiny.census.json').read_bytes()
            mod.run('mock',PLAN,out,resume=True)
            self.assertEqual((out/'tiny.census.json').read_bytes(),original)


if __name__=='__main__':unittest.main()
