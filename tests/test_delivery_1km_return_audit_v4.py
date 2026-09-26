"""Numerical QP retry must keep the original feasibility gate."""

import unittest
from unittest.mock import patch

import numpy as np

from delivery_1km.return_audit_v4 import RetryController
from nav3d_v2.controller import SegmentTrackingController


class ReturnQPTests(unittest.TestCase):
    def test_retry_solves_and_checks_original_constraints(self) -> None:
        rows = np.eye(3)
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        with patch.object(SegmentTrackingController, "_solve",
                          return_value=(None, 0.0, "maximum iterations reached")):
            solution, _, reason = RetryController._solve(
                np.array([2.0, 0.0, 0.0]), rows, lower, upper,
            )
        self.assertIsNotNone(solution)
        self.assertTrue(reason.startswith("retry_solved"))
        self.assertTrue(np.all(rows @ solution <= upper + 2e-5))
        self.assertTrue(np.all(rows @ solution >= lower - 2e-5))

    def test_retry_does_not_accept_infeasible_qp(self) -> None:
        rows = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        lower = np.array([1.0, -np.inf])
        upper = np.array([np.inf, -1.0])
        with patch.object(SegmentTrackingController, "_solve",
                          return_value=(None, 0.0, "maximum iterations reached")):
            solution, _, reason = RetryController._solve(
                np.zeros(3), rows, lower, upper,
            )
        self.assertIsNone(solution)
        self.assertTrue(reason.startswith("retry_"))


if __name__ == "__main__":
    unittest.main()
