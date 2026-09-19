import unittest
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import sys
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from fpl.problem import load_problem, Operation
from fpl.protocols import enumerate_protocols, validate_protocol, PlanningLimit
from fpl.belief import PlannerState, ObservationHistory, update, outcomes
from fpl.environment import Environment, PrivateTruth
from fpl.teachers.exact_bayes import ExactBayes, known_model_value
from fpl.policies.channel_cover import ChannelCover


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.p = load_problem(ROOT/"configs/fixture.json")

    def test_four_cells_and_negative_control(self):
        catalogues = {}
        for capacity in (3, 4):
            for on in (False, True):
                p = replace(self.p, capacity=capacity, max_measurements=2 if on else 1)
                routes = enumerate_protocols(p, p.budget)
                catalogues[capacity, on] = {r.operations for r in routes}
                self.assertEqual(len(routes), 6 if capacity == 4 and on else 4)
                for route in routes:
                    self.assertEqual(validate_protocol(p, route.operations, p.budget), route)
        self.assertEqual(catalogues[3, True], catalogues[4, False])
        self.assertEqual(catalogues[3, True], catalogues[3, False])
        self.assertLess(catalogues[3, True], catalogues[4, True])

    def test_debit_precedes_reload_and_independent_time(self):
        p = replace(self.p, operations=(Operation("dock", "q", "q", 3, 5, ("dock",)),))
        with self.assertRaises(ValueError):
            validate_protocol(p, ("dock",), 6)
        self.assertEqual(enumerate_protocols(p, 6), ())
        p = replace(p, capacity=5)
        plant = Environment(p, PrivateTruth(0, 0))
        result = plant.execute(("dock",))
        self.assertEqual(result.released_at, 3)
        self.assertEqual(result.operations[0]["resource_after_debit"], 0)
        self.assertEqual(result.operations[0]["resource_after"], 5)
        self.assertEqual(plant.planner_state().remaining, 3)

    def test_no_partial_terminal_or_mid_batch_reset(self):
        for names in (("undock",), ("dock", "dock"), ("return_A",)):
            with self.assertRaises(ValueError):
                validate_protocol(self.p, names, 6)

    def test_release_history_posterior_and_no_truth_in_state(self):
        a = Environment(self.p, PrivateTruth(0, 0))
        b = Environment(self.p, PrivateTruth(2, 999))
        self.assertEqual(a.planner_state(), b.planner_state())
        policy = ChannelCover(self.p)
        self.assertEqual(policy.select(a.planner_state()), policy.select(b.planner_state()))
        result = a.execute(("undock", "measure_A", "A_to_B", "return_B"))
        self.assertEqual(result.released_at, 4)
        self.assertEqual(a.planner_state().history.released, result.feedback)
        self.assertEqual(a.planner_state().posterior, update(self.p, self.p.prior, result.feedback))
        self.assertEqual(set(vars(a.planner_state())), {"posterior", "remaining", "resource", "history"})

    def test_bayes_update_and_probability_mass(self):
        posterior = update(self.p, self.p.prior, (("A", 1),))
        self.assertEqual(posterior, (F(1,13), F(6,13), F(6,13)))
        self.assertEqual(sum(m for _, m, _ in outcomes(self.p, self.p.prior, ("A", "B"))), 1)
        self.assertEqual(update(self.p, self.p.prior, (("dock", 1),)), self.p.prior)

    def test_channel_coverage_bundles_without_redundant_path_coverage(self):
        policy = ChannelCover(self.p)
        initial = PlannerState(self.p.prior, 6, 4)
        self.assertEqual(set(policy.select(initial).channels), {"A", "B"})
        # Once both informative channels have been seen, exploit posterior rate.
        state = PlannerState((F(1), F(0), F(0)), 6, 4,
                             ObservationHistory((("A",0),("B",0))))
        self.assertEqual(policy.select(state).channels, ("dock",))

    def test_known_model_and_bayes_distinction(self):
        p = replace(self.p, budget=12)
        for cap in (3, 4):
            cell = replace(p, capacity=cap)
            self.assertEqual(tuple(known_model_value(cell, i) for i in range(3)),
                             (F(3,5), F(6,5), F(19,5)))
        value = ExactBayes(self.p).value(6, self.p.prior)
        informed = sum(w*known_model_value(self.p,i) for i,w in enumerate(self.p.prior))
        self.assertLess(value, informed)
        for i in range(3):
            delta = tuple(F(int(i == j)) for j in range(3))
            self.assertEqual(ExactBayes(self.p).value(6, delta), known_model_value(self.p,i))

    def test_limits_report_unresolved(self):
        with self.assertRaises(PlanningLimit):
            ExactBayes(self.p, max_states=1).value(6, self.p.prior)
        with self.assertRaises(PlanningLimit):
            enumerate_protocols(self.p, 6, max_nodes=1)
        with self.assertRaises(PlanningLimit):
            list(outcomes(self.p, self.p.prior, ("A", "B"), max_outcomes=1))

    def test_arbitrary_hypothesis_and_channel_counts(self):
        p = replace(self.p, channels=("C",), hypotheses=((F(1,4),),(F(3,4),)),
                    prior=(F(1,2),F(1,2)), operations=(Operation("sample","q","q",2,1,("C",)),))
        self.assertEqual(ExactBayes(p).value(6,p.prior), F(3,2))
        self.assertEqual(known_model_value(p,1), F(9,4))

    def test_zero_energy_cycles_are_time_bounded(self):
        p = replace(self.p, operations=(Operation("out","q","prepared",1,0),
                    Operation("loop","prepared","prepared",1,0),
                    Operation("back","prepared","q",1,0)))
        self.assertEqual(len(enumerate_protocols(p,6)), 5)

    def test_frozen_committed_solver_regression(self):
        # Read-only code comparison; no original artifacts or truth files opened.
        source = ROOT.parents[1]/"scripts/audit_resource_separation_bridge.py"
        spec = importlib.util.spec_from_file_location("frozen_reference", source)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for capacity in (3, 4):
            p = replace(self.p, capacity=capacity)
            risks = old.committed_terminal(6, p.prior, capacity)
            old_value = sum(w*(6*g-r) for w,g,r in zip(p.prior, old.GAINS, risks))
            self.assertEqual(ExactBayes(p).value(6,p.prior), old_value)

    def test_budget_changes_feasible_decision(self):
        planner = ExactBayes(self.p)
        short = planner.select(PlannerState(self.p.prior,1,4))
        longer = planner.select(PlannerState(self.p.prior,4,4))
        self.assertEqual(short.channels, ("dock",))
        self.assertNotEqual(longer.channels, short.channels)

    def test_reject_unknown_contract_and_bad_probabilities(self):
        with self.assertRaises(ValueError):
            replace(self.p, terminal_rule="arbitrary")
        with self.assertRaises(ValueError):
            replace(self.p, prior=(F(1),F(1),F(1)))


if __name__ == "__main__":
    unittest.main()
