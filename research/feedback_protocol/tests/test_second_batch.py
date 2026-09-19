import json
import unittest
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fpl.problem import PublicProblem,Operation,UtilitySpec,load_problem
from fpl.generator import generate,problem_json
from fpl.registry import Registry,structural_key
from fpl.dataset import build
from fpl.environment import Environment,PrivateTruth
from fpl.belief import PlannerState
from fpl.utility import protocol_vector
from fpl.protocols import enumerate_protocols,validate_protocol,PlanningLimit
from fpl.teachers.exact_bayes import ExactBayes
from fpl.teachers.minimax_small import solve
from fpl.policies.beam_bayes import BeamBayes
from fpl.policies.posterior_sampling import PosteriorSampling


def task_problem(budget=1,probe=False):
    ops = [Operation("left","q","q",2 if probe else 1,1,utility=UtilitySpec(by_hypothesis=(F(1),F(0)))),
           Operation("right","q","q",2 if probe else 1,1,utility=UtilitySpec(by_hypothesis=(F(0),F(1))))]
    if probe:
        ops.append(Operation("sense","q","q",1,1,("c",),UtilitySpec()))
    return PublicProblem("decision","decision","debug-only",("q",),"q",("c",),
                         ((F(0),),(F(1),)),(F(9,10),F(1,10)),tuple(ops),1,budget,1,
                         objective="cumulative_task_utility")


class SecondBatchTests(unittest.TestCase):
    def test_score_is_not_an_observation(self):
        p = task_problem()
        a,b = Environment(p,PrivateTruth(0,0)),Environment(p,PrivateTruth(1,0))
        self.assertNotEqual(a.execute(("left",)).reward,b.execute(("left",)).reward)
        self.assertEqual(a.planner_state(),b.planner_state())

    def test_information_without_immediate_utility(self):
        p = task_problem(3,True)
        plant = Environment(p,PrivateTruth(1,0))
        out = plant.execute(("sense",))
        self.assertEqual(out.reward,0)
        self.assertEqual(plant.planner_state().posterior,(F(0),F(1)))
        planner = ExactBayes(p)
        self.assertEqual(planner.select(PlannerState(p.prior,3,1)).operations,("sense",))
        self.assertEqual(planner.value(3,p.prior),1)

    def test_exact_minimax_randomization_and_duality(self):
        p = task_problem()
        result = solve(p)
        self.assertEqual(result["value"],F(1,2))
        self.assertEqual(result["risks"],(F(1,2),)*2)
        self.assertEqual(sum(w for w,_ in result["mixture"]),1)
        self.assertEqual(result["least_favorable_prior"],(F(1,2),)*2)
        self.assertEqual(ExactBayes(p).value(1,p.prior),F(9,10))

    def test_minimax_feedback_tree_and_limits(self):
        p = task_problem(3,True)
        result = solve(p)
        self.assertEqual(result["value"],0)
        for w,tree in result["mixture"]:
            self.assertEqual(tree.route.operations,("sense",))
            self.assertEqual(len(tree.branches),2)
        with self.assertRaises(PlanningLimit):
            solve(p,max_combinations=1)

    def test_family_and_clone_split_enforcement(self):
        p = generate(3,3,split="train")
        registry = Registry()
        registry.register(p)
        registry.register(replace(p,capacity=7,budget=12))
        with self.assertRaises(ValueError):
            registry.register(replace(p,split_id="test",family_id="renamed",hypotheses=p.hypotheses[::-1]))
        with self.assertRaises(ValueError):
            registry.register(generate(5,4,split="dev"))
        rename = {n:"renamed_"+n for n in p.nodes}
        clone = replace(p,nodes=tuple(rename[n] for n in p.nodes),reset=rename[p.reset],
                        operations=tuple(replace(o,name="x"+o.name,source=rename[o.source],target=rename[o.target]) for o in p.operations))
        self.assertEqual(structural_key(p),structural_key(clone))
        self.assertEqual(Registry.from_dict(registry.to_dict()).to_dict(),registry.to_dict())

    def test_generation_plan_and_variation(self):
        tasks,registry = build(json.loads((ROOT/"configs/families/splits.json").read_text()))
        self.assertEqual(len(tasks),20)
        self.assertEqual(len(registry.entries),20)
        paired = generate(4,3,sharing="paired")
        self.assertTrue(all(row[0] == row[1] and row[2] == row[3] for row in paired.hypotheses))
        self.assertEqual(generate(seed=7).instance_hash,generate(seed=7).instance_hash)
        self.assertNotEqual(generate(seed=7).instance_hash,generate(seed=8).instance_hash)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"task.json"
            path.write_text(problem_json(paired))
            self.assertEqual(load_problem(path),paired)

    def test_planners_never_call_full_enumeration(self):
        p = task_problem(3,True)
        state = PlannerState(p.prior,3,1)
        with patch("fpl.protocols.enumerate_protocols",side_effect=AssertionError("full enumeration")):
            planner = BeamBayes(p,width=8,depth=2)
            route = planner.select(state)
            self.assertEqual(route.operations,("sense",))
            validate_protocol(p,route.operations,3)
            ps = PosteriorSampling(p,seed=1)
            validate_protocol(p,ps.select(state).operations,3)

    def test_protocol_combinatorics(self):
        # Sum of ordered partial permutations, plus empty and hypothesis tasks.
        from math import factorial
        for m in (2,3,4):
            p = generate(m,3,budget=m+2)
            expected = 4+sum(factorial(m)//factorial(m-k) for k in range(1,m+1))
            self.assertEqual(len(enumerate_protocols(p,p.budget)),expected)
            star = generate(m,3,"star",budget=m+2)
            self.assertEqual(len(enumerate_protocols(star,star.budget)),m+4)


if __name__ == "__main__":
    unittest.main()
