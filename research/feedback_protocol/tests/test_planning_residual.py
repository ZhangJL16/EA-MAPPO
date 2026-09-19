import importlib.util
import json
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
from fpl.problem import PublicProblem,Operation,UtilitySpec
from fpl.protocols import validate_protocol
from fpl.belief import PlannerState
from fpl.teacher_data import exact_label,state_key
from fpl_v2.generator import as_json

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('residual',ROOT/'scripts/planning_residual.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
PLAN=json.loads((ROOT/'configs/planning_residual_v1.json').read_text())


def fixture():
    return PublicProblem('tiny','tiny','train',('q',),'q',('c',),((F(0),),(F(1),)),(F(1,2),F(1,2)),
        (Operation('sense','q','q',1,1,('c',),UtilitySpec()),
         Operation('task0','q','q',1,1,(),UtilitySpec(by_hypothesis=(F(1),F(0)))),
         Operation('task1','q','q',1,1,(),UtilitySpec(by_hypothesis=(F(0),F(1))))),
        1,3,1,objective='cumulative_task_utility')


class Fixed:
    def __init__(self,p,stop=False):self.p,self.stop,self.calls=p,stop,0;self.expansions=self.model_calls=0;self.stats={}
    def select(self,s):
        self.calls+=1
        if self.stop:return None
        return validate_protocol(self.p,('sense' if self.calls==1 else 'task0',),s.remaining)


class ResidualTests(unittest.TestCase):
    def test_branch_state_and_exact_identity(self):
        p=fixture();labels=mod.Labels(p,[],PLAN)
        result=mod.decompose(p,Fixed(p),labels,PLAN)
        self.assertEqual(result['status'],'exact');self.assertEqual(result['value_star'],'2')
        self.assertEqual(result['value_policy'],'1');self.assertEqual(result['gap'],'1')
        self.assertEqual(result['weighted_residual_sum'],'1')
        self.assertEqual(result['rows'][0]['residual'],'0')
        self.assertEqual(len(result['rows']),5)
        self.assertTrue(all(r['action']==['task0'] for r in result['rows'][1:]))

    def test_stop_residual_not_zero(self):
        p=fixture();result=mod.decompose(p,Fixed(p,True),mod.Labels(p,[],PLAN),PLAN)
        self.assertEqual(result['gap'],'2');self.assertEqual(len(result['rows']),1)
        self.assertEqual(result['rows'][0]['chosen_type'],'stop')

    def test_unresolved_not_zero_and_q_completeness(self):
        p=fixture();state=PlannerState(p.prior,3,1)
        with self.assertRaises(mod.PlanningLimit):mod.decision(p,state,dict(status='unresolved'),None)
        label=exact_label(p,state,PLAN['supplemental_label_limits'])
        self.assertEqual(mod.decision(p,state,label,validate_protocol(p,('sense',),3))['residual'],'0')
        self.assertGreater(F(mod.decision(p,state,label,validate_protocol(p,('task0',),3))['residual']),0)

    def test_resume_scope_and_receipts(self):
        p=fixture();state=PlannerState(p.prior,3,1)
        shard=dict(problem=as_json(p),metadata={'root_seed':1},records=[dict(id=state_key(state),
            state=dict(posterior=list(map(str,p.prior)),remaining=3,resource=1),
            label=exact_label(p,state,PLAN['supplemental_label_limits']))])
        with TemporaryDirectory() as d,patch.object(mod,'load',return_value=[shard]):
            out=Path(d)/'study';mod.run('mock',PLAN,out,max_new=1)
            raw=(out/'tiny.residual.json').read_bytes();mod.run('mock',PLAN,out,resume=True)
            self.assertEqual(raw,(out/'tiny.residual.json').read_bytes())


if __name__=='__main__':unittest.main()
