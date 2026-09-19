import json
import random
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from fpl.budget import SearchBudget
from fpl.protocols import PlanningLimit,validate_protocol
from fpl.registry import RegistryV2,structural_key
from fpl.scientific_dataset import random_problem,build,freeze
from fpl.environment import Environment,PrivateTruth
from fpl.belief import PlannerState
from fpl.problem import load_problem
from fpl.teachers.minimax_small import certify_matrix,UnresolvedCertificate
from fpl.policies.beam_bayes import BeamBayes
from fpl.policies.one_step_voi import OneStepVOI
from fpl.policies.posterior_sampling import PosteriorSampling
from fpl.policies.channel_cover import ChannelCover
from fpl.evaluation import run,read_rows,ReferencePolicy,references,encode_tree
from fpl.teachers.minimax_small import solve
from fpl.analyze_dev import summarize_cell,analyze
from test_second_batch import task_problem


class ThirdBatchTests(unittest.TestCase):
    def test_independent_lp_supports(self):
        # 2 policies required, but one hypothesis is already least favorable.
        result = certify_matrix(((0,2,1),(2,0,1)))
        self.assertEqual(result["value"],1)
        self.assertEqual(len(result["weights"]),2)
        self.assertEqual(sum(w>0 for w in result["least_favorable_prior"]),1)
        reverse = certify_matrix(((0,2),(2,0),(1,1)))
        self.assertEqual(reverse["value"],1)
        self.assertEqual(len(reverse["weights"]),1)
        self.assertEqual(sum(w>0 for w in reverse["least_favorable_prior"]),2)

    def test_random_matrices_against_scipy(self):
        try:
            from scipy.optimize import linprog
        except ImportError:
            self.skipTest("SciPy is a test-only optional cross-check")
        rng = random.Random(703)
        for n in range(1,5):
            for m in range(1,6):
                for _ in range(3):
                    columns = tuple(tuple(rng.randint(-3,5) for _ in range(n)) for _ in range(m))
                    exact = certify_matrix(columns)
                    lp = linprog([0]*m+[1],A_ub=[[columns[c][r] for c in range(m)]+[-1] for r in range(n)],
                                 b_ub=[0]*n,A_eq=[[1]*m+[0]],b_eq=[1],bounds=[(0,None)]*m+[(None,None)],method="highs")
                    self.assertTrue(lp.success)
                    self.assertAlmostEqual(float(exact["value"]),lp.fun,places=9)
                    self.assertEqual(sum(w for _,w in exact["weights"]),1)
                    self.assertEqual(sum(exact["least_favorable_prior"]),1)
        with self.assertRaises(UnresolvedCertificate):
            certify_matrix(((0,2),(2,0)),max_candidates=1)

    def test_registry_v2_and_scientific_distribution(self):
        plan = json.loads((ROOT/"configs/families/scientific_v1.json").read_text())
        tasks,reg,_ = build(plan)
        self.assertEqual(len(tasks),44)
        self.assertEqual(RegistryV2.from_dict(reg.to_dict()).to_dict(),reg.to_dict())
        shared = [r for r in reg.entries.values() if r["distribution_id"]=="directed-m3-k2-p05"]
        self.assertEqual({r["split"] for r in shared},{"train","dev","test"})
        self.assertEqual({r["ood_axis"] for r in reg.entries.values() if r["split"]=="ood"},
                         {"size","topology","cost_ratio","hypothesis_count"})
        p = tasks[0]
        with self.assertRaises(ValueError):
            reg.register(replace(p,split_id="test"),distribution_id="same",template_id="directed",parameter_seed=0,structure_seed=0)
        a = random_problem(structure_seed=1,parameter_seed=3)
        b = random_problem(structure_seed=1,parameter_seed=4)
        self.assertEqual(structural_key(a),structural_key(b))
        self.assertNotEqual(a.hypotheses,b.hypotheses)

    def test_global_budget_not_per_recursive_call(self):
        p = task_problem(3,True)
        limits = dict(max_expansions=30,max_model_calls=50,max_seconds=10)
        for policy in (BeamBayes(p,budget_limits=limits),OneStepVOI(p,budget_limits=limits),
                       ChannelCover(p,budget_limits=limits),PosteriorSampling(p,budget_limits=limits)):
            route = policy.select(PlannerState(p.prior,3,1))
            if route:
                validate_protocol(p,route.operations,3)
            self.assertLessEqual(policy.stats["expansions"],30)
            self.assertLessEqual(policy.stats["model_calls"],50)
        budget = SearchBudget(max_expansions=1)
        budget.consume(expansions=1)
        with self.assertRaises(PlanningLimit):
            budget.consume(expansions=1)
        with self.assertRaises(ValueError):
            budget.consume(model_calls=-1)

    def test_voi_information_without_reward_no_catalogue(self):
        p = task_problem(3,True)
        with patch("fpl.protocols.enumerate_protocols",side_effect=AssertionError("catalogue")):
            policy = OneStepVOI(p)
            self.assertEqual(policy.select(PlannerState(p.prior,3,1)).operations,("sense",))

    def test_minimax_executor_uses_only_released_feedback(self):
        p = task_problem(3,True)
        certificate = solve(p)
        reference = {"mixture":[dict(weight=str(w),tree=encode_tree(tree)) for w,tree in certificate["mixture"]]}
        for theta in (0,1):
            policy = ReferencePolicy(p,"certified_minimax",reference,0)
            env = Environment(p,PrivateTruth(theta,0),crn_key="test")
            first = policy.select(env.planner_state())
            self.assertEqual(first.operations,("sense",))
            env.execute(first.operations)
            second = policy.select(env.planner_state())
            result = env.execute(second.operations)
            self.assertEqual(result.reward,1)

    def test_crn_channel_query_order_and_private_utility(self):
        p = random_problem(structure_seed=2,parameter_seed=3,capacity=5,budget=12)
        for seed in range(10):
            left = Environment(p,PrivateTruth(0,seed),crn_key="paired-instance")
            right = Environment(p,PrivateTruth(0,seed),crn_key="paired-instance")
            la = left.execute(("prepare","probe0","return0")).feedback
            lb = left.execute(("prepare","probe1","return1")).feedback
            rb = right.execute(("prepare","probe1","return1")).feedback
            ra = right.execute(("prepare","probe0","return0")).feedback
            self.assertEqual(la,ra)
            self.assertEqual(lb,rb)
            before = right.planner_state().posterior
            right.execute(("task0",))
            self.assertEqual(before,right.planner_state().posterior)

    def test_worst_risk_is_max_of_means(self):
        rows = []
        for h,values in enumerate(((0,10),(6,6))):
            for ns,x in enumerate(values):
                rows.append(dict(status="completed",theta=h,noise_seed=ns,policy_seed=0,
                    sample_shortfall=str(x),utility="0",planning_seconds=0,expansions=0,model_calls=0,
                    measurements=0,truncated_selections=0,protocol_lengths=[]))
        result = summarize_cell(rows,(F(1,2),)*2)
        self.assertEqual(result["worst_hypothesis_risk"],6)
        self.assertEqual(result["bayes_risk"],5.5)

    def test_runner_seal_resume_and_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = json.loads((ROOT/"configs/families/scientific_v1.json").read_text())
            plan["groups"] = [dict(plan["groups"][1],roots=1,budgets=[4],capacities=[3])]
            (tmp/"data-plan.json").write_text(json.dumps(plan))
            freeze(tmp/"data-plan.json",tmp/"data")
            config = json.loads((ROOT/"configs/dev_pilot_v1.json").read_text())
            config.update(noise_seeds=[0,1],policy_seeds=[0],methods=["channel_cover"],tiers={"tiny":config["tiers"]["small"]})
            config["references"]["max_combinations"] = 1
            (tmp/"run-plan.json").write_text(json.dumps(config))
            paused = run(tmp/"data",tmp/"run-plan.json",tmp/"run",max_new_episodes=1)
            self.assertEqual(paused["status"],"paused")
            finished = run(tmp/"data",tmp/"run-plan.json",tmp/"run",resume=True)
            self.assertEqual(finished["rows"],12)
            self.assertEqual(run(tmp/"data",tmp/"run-plan.json",tmp/"run",resume=True)["added"],0)
            result = analyze(tmp/"data",tmp/"run")
            self.assertEqual(result["episode_rows"],12)
            self.assertTrue(any(r["status"]=="unresolved" for r in result["cells"]))
            config["noise_seeds"] = [0]
            (tmp/"run-plan.json").write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                run(tmp/"data",tmp/"run-plan.json",tmp/"run",resume=True)
            with (tmp/"run"/"episodes.jsonl").open("a") as f:
                f.write("{")
            with self.assertRaises(ValueError):
                read_rows(tmp/"run"/"episodes.jsonl")


if __name__=="__main__":
    unittest.main()
