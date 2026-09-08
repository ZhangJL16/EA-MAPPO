from __future__ import annotations

import unittest

import numpy as np

from experiments.rechargeability_safety.dual_viability import (
    budget_labels,
    make_job_keys,
    paired_semantic_summary,
)


class DualViabilityTests(unittest.TestCase):
    def test_budget_labels_are_monotone_and_fail_closed(self) -> None:
        labels = budget_labels(
            structurally_safe=np.asarray([True, True, False]),
            energy_fraction=np.asarray([0.2, 0.5, 0.1]),
            budgets=np.asarray([0.1, 0.3, 0.6]),
        )
        np.testing.assert_array_equal(
            labels,
            np.asarray(
                [[False, True, True], [False, False, True], [False, False, False]]
            ),
        )
        self.assertTrue(np.all(labels[:, 1:] >= labels[:, :-1]))

    def test_jobs_use_only_task_anchors(self) -> None:
        anchors = [
            {
                "anchor_id": "a",
                "leg": 0,
                "candidate_names": ["nominal", "brake"],
                "candidate_actions": [[0, 0, 0], [1, 0, 0]],
            },
            {
                "anchor_id": "b",
                "leg": 1,
                "candidate_names": ["nominal"],
                "candidate_actions": [[0, 0, 0]],
            },
        ]
        self.assertEqual(
            make_job_keys(anchors),
            [("a", "return_now"), ("a", "option/nominal"), ("a", "option/brake")],
        )

    def test_paired_direction_counts_and_action_variation(self) -> None:
        report = paired_semantic_summary(
            anchor_ids=np.asarray(["a", "a", "b", "b"]),
            candidate_names=np.asarray(["n", "b", "n", "b"]),
            return_now=np.asarray([[1], [1], [0], [0]], dtype=bool),
            option=np.asarray([[1], [0], [1], [1]], dtype=bool),
            completion=np.asarray([[0], [1], [1], [1]], dtype=bool),
            budgets=np.asarray([0.4]),
        )[0]
        self.assertEqual(report["completion_yes_option_no"], 1)
        self.assertEqual(report["completion_no_option_yes"], 1)
        self.assertEqual(report["return_now_yes_option_no"], 1)
        self.assertEqual(report["return_now_no_option_yes"], 2)
        self.assertEqual(report["mixed_option_anchors"], 1)


if __name__ == "__main__":
    unittest.main()
