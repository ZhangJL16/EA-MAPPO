import copy
from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fpl.teacher_data import (exact_label, audit_label, collect_states, build_tasks,
                             run, draw_index, state_key)
from fpl.belief import PlannerState
from fpl.evaluation import digest, save_new
from fpl.registry import structural_key
from fpl.difficulty import hierarchical
from prepare_teacher import default_plan
from test_second_batch import task_problem


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.exclusions = dict(forbidden_seeds=[2201, 2202], clone_hashes=[])
        self.plan = default_plan(self.exclusions)
        self.conf = self.plan["solver"]

    def test_exact_q_and_stop(self):
        p = task_problem(3, True)
        label = exact_label(p, PlannerState(p.prior, 3, p.capacity), self.conf)
        self.assertEqual(label["status"], "exact")
        self.assertEqual(F(label["value"]), 1)
        self.assertEqual(label["optimal_protocol"], ["sense"])
        self.assertEqual({tuple(c["operations"] or ()): F(c["q"]) for c in label["candidates"]},
                         {(): F(0), ("sense",): F(1), ("left",): F(9, 10), ("right",): F(1, 10)})
        terminal = exact_label(p, PlannerState(p.prior, 0, p.capacity), self.conf)
        self.assertEqual(terminal["optimal_set"], [None])

    def test_no_partial_label_on_cap(self):
        label = exact_label(task_problem(3, True), PlannerState((F(9, 10), F(1, 10)), 3, 1),
                            dict(self.conf, max_states=1))
        self.assertEqual(label["status"], "unresolved")
        self.assertNotIn("value", label)
        self.assertNotIn("candidates", label)

    def test_mixed_histories_and_arithmetic(self):
        p = replace(task_problem(3, True), split_id="train")
        states, logs = collect_states(p, 100001, self.plan)
        self.assertEqual({x["source"] for x in logs}, {"exact", "random", "voi", "beam", "posterior_sampling"})
        self.assertTrue(any(s.posterior != p.prior for s, _ in states.values()))
        self.assertTrue(any(s.posterior == (F(1), F(0)) for s, _ in states.values()))
        self.assertTrue(any(s.posterior == (F(0), F(1)) for s, _ in states.values()))
        for key, (state, provenance) in states.items():
            row = dict(state=dict(posterior=list(map(str, state.posterior)), remaining=state.remaining,
                                  resource=state.resource), provenance=provenance,
                       label=exact_label(p, state, self.conf))
            audit_label(p, row)
            bad = copy.deepcopy(row)
            bad["label"]["candidates"][0]["q"] = "1"
            with self.assertRaises(ValueError):
                audit_label(p, bad)

    def test_seed_and_clone_exclusion(self):
        plan = dict(self.plan, roots_per_condition=1, max_structure_attempts=2,
                    conditions=self.plan["conditions"][:1])
        jobs, _ = build_tasks(plan, self.exclusions)
        ex = dict(self.exclusions, clone_hashes=[structural_key(jobs[0][0])])
        plan["exclusions_hash"] = digest(ex)
        jobs, rejected = build_tasks(plan, ex)
        self.assertEqual(jobs, [])
        self.assertTrue(rejected[-1]["unresolved"])
        with self.assertRaises(ValueError):
            build_tasks(dict(plan, root_start=2201), ex)
        with self.assertRaises(ValueError):
            build_tasks(dict(plan, split="dev"), ex)

    def test_exact_draw_and_no_truth_access(self):
        self.assertEqual(draw_index([F(0), F(1)], random.Random(1)), 1)
        p = replace(task_problem(3, True), split_id="train")
        with patch("fpl.environment.Environment", side_effect=AssertionError("private environment forbidden")):
            a, _ = collect_states(p, 9, self.plan)
            b, _ = collect_states(p, 9, self.plan)
        self.assertEqual(a, b)

    def test_resume_seal_and_corruption(self):
        plan = dict(self.plan, conditions=self.plan["conditions"][:1], roots_per_condition=2,
                    max_rollout_batches=1, states_per_source=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_new(root / "plan.json", plan)
            save_new(root / "exclusions.json", self.exclusions)
            args = (root / "plan.json", root / "exclusions.json", root / "data")
            a = run(*args, max_new=1)
            self.assertEqual(a["completed_roots"], 1)
            checkpoint = next((root / "data").glob("*.shard.json"))
            original = checkpoint.read_bytes()
            b = run(*args, resume=True, max_new=1)
            self.assertEqual(b["status"], "complete")
            self.assertEqual(checkpoint.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                run(*args)
            with patch("fpl.teacher_data.source_hashes", return_value={"drift": "bad"}):
                with self.assertRaises(ValueError):
                    run(*args, resume=True)
            envelope = json.loads(checkpoint.read_text())
            envelope["payload"]["cpu_seconds"] = -1
            save_new(checkpoint, envelope)
            with self.assertRaises(ValueError):
                run(*args, resume=True)


if __name__ == "__main__":
    unittest.main()
