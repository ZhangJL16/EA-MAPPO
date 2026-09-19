import copy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fpl.work import PlanningWorkBudget,WorkPolicy,WatchdogExpired
from fpl.teachers.exact_policy import ExactPolicyEvaluator
from fpl.teachers.exact_bayes import ExactBayes
from fpl.belief import PlannerState
from fpl.protocols import Protocol,validate_protocol,PlanningLimit
from fpl.policies.channel_cover import ChannelCover
from fpl.policies.beam_bayes import BeamBayes
from fpl.policies.posterior_sampling import PosteriorSampling
from fpl.policies.one_step_voi import OneStepVOI
from fpl.policies.belief_mcts import BeliefMCTS
from fpl.difficulty import hierarchical,tasks
from fpl.environment import Environment,PrivateTruth
from test_second_batch import task_problem


class FourthBatchTests(unittest.TestCase):
    def test_watchdog_is_not_policy_cutoff(self):
        budget = PlanningWorkBudget(max_expansions=1)
        budget.consume(expansions=1)
        with self.assertRaises(PlanningLimit):
            budget.consume(expansions=1)
        with patch("fpl.work.monotonic",return_value=budget.started+1000):
            with self.assertRaises(WatchdogExpired):
                budget.consume()
        self.assertFalse(issubclass(WatchdogExpired,PlanningLimit))

    def test_policy_evaluation_utility_and_episode_caps(self):
        p = task_problem(3,True)
        for cls in (ChannelCover,BeamBayes,OneStepVOI):
            agent = WorkPolicy(p,cls(p),dict(max_expansions=1000,max_model_calls=10000),
                               dict(max_expansions=2000,max_model_calls=20000))
            result = ExactPolicyEvaluator(p).evaluate(agent)
            self.assertEqual(result["status"],"exact_conditional_policy")
            self.assertEqual(tuple(map(F,result["utility_by_hypothesis"])),(F(1),)*2)
            self.assertLessEqual(result["worst_episode_expansions"],2000)
            self.assertLessEqual(result["worst_episode_model_calls"],20000)

    def test_rng_and_mutable_policy_state_clone(self):
        p = task_problem(3,True)
        agent = WorkPolicy(p,PosteriorSampling(p,seed=4),dict(max_expansions=1000,max_model_calls=10000))
        before = agent.policy.rng.getstate()
        a = ExactPolicyEvaluator(p).evaluate(agent)
        b = ExactPolicyEvaluator(p).evaluate(agent)
        self.assertEqual(before,agent.policy.rng.getstate())
        self.assertEqual(a["utility_by_hypothesis"],b["utility_by_hypothesis"])
        self.assertEqual(a["action_map"],b["action_map"])

    def test_history_not_merged_by_belief(self):
        # Equal posterior does not imply equal internal policy state. Preserve
        # an internal counter after feedback, rather than restarting each node.
        p = task_problem(3,True)
        class TwoStage:
            def __init__(self): self.count=0
            def select(self,state):
                self.count += 1
                if self.count==1: return Protocol(("sense",),1,("c",))
                return Protocol(("left" if state.posterior[0] else "right",),2,())
        result = ExactPolicyEvaluator(p).evaluate(TwoStage())
        self.assertEqual(result["utility_by_hypothesis"],["1","1"])
        limited = ExactPolicyEvaluator(p,max_nodes=1).evaluate(TwoStage())
        self.assertEqual(limited["status"],"unresolved")
        self.assertNotIn("utility_by_hypothesis",limited)

    def test_mcts_legality_and_no_enumeration(self):
        p = hierarchical(root_seed=999,depth=2,budget=8)
        agent = WorkPolicy(p,BeliefMCTS(p,seed=7),dict(max_expansions=100,max_model_calls=1000),
                           dict(max_expansions=120,max_model_calls=1200))
        env = Environment(p,PrivateTruth(0,1))
        with patch("fpl.protocols.enumerate_protocols",side_effect=AssertionError("enumeration")):
            while env.time<p.budget:
                route = agent.select(env.planner_state())
                if route is None: break
                validate_protocol(p,route.operations,p.budget-env.time)
                env.execute(route.operations)
        self.assertLessEqual(agent.expansions,120)
        self.assertLessEqual(agent.model_calls,1200)
        self.assertGreater(agent.policy.stats.get("simulations",0),-1)

    def test_difficulty_axes_are_public_and_predefined(self):
        plan = json.loads((ROOT/"configs/difficulty_dev_v2.json").read_text())
        generated = list(tasks(plan))
        self.assertEqual(len(generated),12)
        self.assertEqual({p.split_id for p,_ in generated},{"dev"})
        p = hierarchical(root_seed=999,depth=2)
        self.assertEqual(len(p.hypotheses),4)
        self.assertEqual(len(p.channels),3)
        self.assertEqual(sum(row[1]==F(1,2) for row in p.hypotheses),2)
        self.assertEqual(sum(row[2]==F(1,2) for row in p.hypotheses),2)
        self.assertEqual(p.instance_hash,hierarchical(root_seed=999,depth=2).instance_hash)

    def test_v4_runner_resume_and_seal(self):
        from fpl.necessity import run
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = json.loads((ROOT/"configs/difficulty_dev_v2.json").read_text())
            plan.update(root_seeds=[998],conditions=[dict(id="test",overrides={"budget":3})],methods=["one_step_voi"],
                        tiers={"medium":plan["tiers"]["medium"]})
            path = tmp/"plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(run(path,tmp/"study",max_new=1)["status"],"paused")
            self.assertEqual(run(path,tmp/"study",resume=True)["completed"],2)
            self.assertEqual(run(path,tmp/"study",resume=True)["completed"],2)
            plan["root_seeds"] = [997]
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                run(path,tmp/"study",resume=True)


if __name__=="__main__":
    unittest.main()
