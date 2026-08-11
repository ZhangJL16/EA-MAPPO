from __future__ import annotations

from itertools import permutations
from pathlib import Path
import unittest

import numpy as np
import torch

from cert_runtime.generator_sac import GeneratorSAC, GeneratorSACConfig
from learned_fields.dual_safety import DualFieldFeatureSpec
from learned_fields.proposal_ranker import ProposalCandidateBatch, TrainingBehaviorProposalRanker


class DualFieldProposalTests(unittest.TestCase):
    def setUp(self) -> None:
        layout = {
            "position": slice(0, 3),
            "velocity": slice(3, 6),
            "energy": slice(6, 7),
            "goal_delta": slice(7, 10),
            "station_delta": slice(10, 13),
            "required_return_energy": slice(13, 14),
            "energy_margin": slice(14, 15),
            "mission_mode": slice(15, 20),
            "lidar_distances": slice(20, 52),
            "lidar_valid": slice(52, 84),
            "recovery_corridor": slice(84, 90),
        }
        self.spec = DualFieldFeatureSpec.from_observation_layout(layout)
        self.observation = np.linspace(-0.4, 0.4, 90, dtype=np.float32)

    def test_feature_whitelist_excludes_goal_energy_mode_and_certificate_surrogates(self) -> None:
        collision = self.spec.collision_features(self.observation, np.zeros(3), np.ones(3))
        energy = self.spec.recovery_energy_features(self.observation)
        changed = self.observation.copy()
        # Apply forbidden perturbations explicitly; none overlaps the whitelist.
        changed[6:10] += 2.0
        changed[13:20] -= 3.0
        changed[84:90] += 4.0
        np.testing.assert_array_equal(
            collision,
            self.spec.collision_features(changed, np.zeros(3), np.ones(3)),
        )
        np.testing.assert_array_equal(energy, self.spec.recovery_energy_features(changed))
        self.assertEqual(collision.shape, (76,))
        self.assertEqual(energy.shape, (9,))

    def test_selected_action_remains_in_original_zonotope(self) -> None:
        latent = np.array(((0.0, 0.0, 0.0), (0.5, -0.3, 0.2), (-0.4, 0.1, 0.7)))
        center = np.array((0.01, -0.02, 0.03))
        generator = np.diag((0.05, 0.04, 0.03))
        action = TrainingBehaviorProposalRanker.map_latent(center, generator, latent)
        candidates = ProposalCandidateBatch(latent, action, np.zeros((3, 9)))
        ranker = TrainingBehaviorProposalRanker(2.0 / 3.0)
        index, audit = ranker.select(
            candidates,
            np.zeros(9),
            lambda observation, actions, successors: (
                np.array((0.1, 0.9, 0.8)),
                np.array((3.0, 9.0, 1.0)),
            ),
            generator_authority=True,
        )
        self.assertEqual(index, 2)
        eta = np.linalg.solve(generator, action[index] - center)
        self.assertTrue(np.all(np.abs(eta) < 1.0))
        self.assertFalse(audit.fallback_to_candidate_zero)

    def test_nan_or_missing_authority_falls_back_to_candidate_zero(self) -> None:
        latent = np.zeros((2, 3))
        candidates = ProposalCandidateBatch(latent, latent, np.zeros((2, 9)))
        ranker = TrainingBehaviorProposalRanker()
        index, audit = ranker.select(
            candidates,
            np.zeros(9),
            lambda observation, actions, successors: (np.array((np.nan, 1.0)), np.zeros(2)),
            generator_authority=True,
        )
        self.assertEqual(index, 0)
        self.assertTrue(audit.fallback_to_candidate_zero)
        index, audit = ranker.select(candidates, np.zeros(9), None, generator_authority=False)
        self.assertEqual(index, 0)
        self.assertEqual(audit.reason, "GENERATOR_AUTHORITY_UNAVAILABLE")

    def test_certificate_modules_do_not_import_learned_fields(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protected = (
            root / "cert_runtime",
            root / "envs" / "certified_uav" / "recoverability.py",
            root / "envs" / "certified_uav" / "mission_certificate.py",
            root / "envs" / "certified_uav" / "recovery_atlas.py",
        )
        for target in protected:
            paths = target.rglob("*.py") if target.is_dir() else (target,)
            for path in paths:
                self.assertNotIn("learned_fields", path.read_text(encoding="utf-8"), msg=str(path))

    def test_shadow_candidate_rng_does_not_advance_main_actor_rng(self) -> None:
        agent = GeneratorSAC(90, GeneratorSACConfig(hidden_dim=8), seed=19)
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        first = agent.sample_u_candidates(self.observation, 4)
        after = torch.random.get_rng_state().clone()
        torch.testing.assert_close(before, after)
        second = agent.sample_u_candidates(self.observation, 4)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.shape, (4, 3))

    def test_ranker_passes_candidate_successors_to_energy_scorer(self) -> None:
        latent = np.zeros((2, 3), dtype=np.float64)
        successors = np.array(((1.0, 2.0), (3.0, 4.0)), dtype=np.float64)
        candidates = ProposalCandidateBatch(latent, latent, successors)

        def score(observation, actions, candidate_successors):
            np.testing.assert_array_equal(candidate_successors, successors)
            return np.array((1.0, 1.0)), candidate_successors[:, 0]

        index, audit = TrainingBehaviorProposalRanker().select(
            candidates,
            np.zeros(2),
            score,
            generator_authority=True,
        )
        self.assertEqual(index, 0)
        self.assertFalse(audit.fallback_to_candidate_zero)

    def test_scalar_monotone_bottleneck_cannot_preserve_crossing_orders(self) -> None:
        collision_order = ("a", "b", "c")
        recovery_order = ("b", "a", "c")
        feasible = []
        for scalar_order in permutations(("a", "b", "c")):
            monotone_orders = (scalar_order, tuple(reversed(scalar_order)))
            if collision_order in monotone_orders and recovery_order in monotone_orders:
                feasible.append(scalar_order)
        self.assertEqual(feasible, [])
        self.assertNotEqual(recovery_order, collision_order)
        self.assertNotEqual(recovery_order, tuple(reversed(collision_order)))

    def test_finite_rank_recovery_residual_error_bound(self) -> None:
        rng = np.random.default_rng(31)
        max_rank = 12
        epsilon = 0.07
        epsilon_0 = 0.03
        one_step = rng.uniform(0.0, 2.0, size=max_rank + 1)
        one_step[0] = 0.0
        exact = np.zeros(max_rank + 1)
        estimate = np.zeros(max_rank + 1)
        estimate[0] = -epsilon_0
        for rank in range(1, max_rank + 1):
            exact[rank] = one_step[rank] + exact[rank - 1]
            residual = rng.uniform(-epsilon, epsilon)
            estimate[rank] = one_step[rank] + estimate[rank - 1] + residual
            self.assertLessEqual(
                abs(estimate[rank] - exact[rank]),
                epsilon_0 + rank * epsilon + 1e-12,
            )


if __name__ == "__main__":
    unittest.main()
