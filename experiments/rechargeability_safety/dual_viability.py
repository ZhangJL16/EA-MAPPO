"""Pure utilities for the return-now/option-preservation diagnostic."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def budget_labels(*, structurally_safe: np.ndarray, energy_fraction: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    safe = np.asarray(structurally_safe, dtype=np.bool_).reshape(-1)
    energy = np.asarray(energy_fraction, dtype=np.float64).reshape(-1)
    levels = np.asarray(budgets, dtype=np.float64).reshape(-1)
    if safe.size == 0 or safe.shape != energy.shape:
        raise ValueError("safe and energy arrays must be nonempty and aligned")
    if levels.size == 0 or np.any(~np.isfinite(levels)) or np.any(np.diff(levels) < 0.0):
        raise ValueError("budgets must be finite and nondecreasing")
    if np.any(~np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("energy fractions must be finite and nonnegative")
    return safe[:, None] & (energy[:, None] <= levels[None, :] + 1e-12)


def make_job_keys(anchors: Iterable[dict[str, object]]) -> list[tuple[str, str]]:
    """Return deterministic (anchor, arm) keys for task-leg anchors only."""

    keys: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in anchors:
        if int(anchor["leg"]) != 0:
            continue
        anchor_id = str(anchor["anchor_id"])
        if anchor_id in seen:
            raise ValueError(f"duplicate task anchor: {anchor_id}")
        seen.add(anchor_id)
        names = [str(value) for value in anchor["candidate_names"]]
        actions = list(anchor["candidate_actions"])
        if len(names) != len(actions) or len(set(names)) != len(names):
            raise ValueError(f"invalid candidate table for {anchor_id}")
        keys.append((anchor_id, "return_now"))
        keys.extend((anchor_id, f"option/{name}") for name in names)
    return keys


def paired_semantic_summary(
    *,
    anchor_ids: np.ndarray,
    candidate_names: np.ndarray,
    return_now: np.ndarray,
    option: np.ndarray,
    completion: np.ndarray,
    budgets: np.ndarray,
) -> list[dict[str, object]]:
    """Summarize paired label directions without inferential overclaiming."""

    anchors = np.asarray(anchor_ids).reshape(-1)
    names = np.asarray(candidate_names).reshape(-1)
    now = np.asarray(return_now, dtype=np.bool_)
    opt = np.asarray(option, dtype=np.bool_)
    old = np.asarray(completion, dtype=np.bool_)
    levels = np.asarray(budgets, dtype=np.float64).reshape(-1)
    if now.shape != opt.shape or opt.shape != old.shape:
        raise ValueError("paired label matrices must align")
    if now.shape != (anchors.size, levels.size) or names.size != anchors.size:
        raise ValueError("identifier and budget dimensions do not align")
    by_anchor: dict[str, list[int]] = defaultdict(list)
    for index, anchor in enumerate(anchors.tolist()):
        by_anchor[str(anchor)].append(index)
    reports: list[dict[str, object]] = []
    for column, budget in enumerate(levels):
        old_yes_new_no = int(np.sum(old[:, column] & ~opt[:, column]))
        old_no_new_yes = int(np.sum(~old[:, column] & opt[:, column]))
        now_yes_action_no = int(np.sum(now[:, column] & ~opt[:, column]))
        now_no_action_yes = int(np.sum(~now[:, column] & opt[:, column]))
        mixed = 0
        for indices in by_anchor.values():
            values = opt[np.asarray(indices), column]
            mixed += int(np.any(values) and not np.all(values))
        reports.append(
            {
                "budget": float(budget),
                "pairs": int(anchors.size),
                "anchors": int(len(by_anchor)),
                "completion_positive": int(np.sum(old[:, column])),
                "option_positive": int(np.sum(opt[:, column])),
                "return_now_positive_repeated": int(np.sum(now[:, column])),
                "completion_yes_option_no": old_yes_new_no,
                "completion_no_option_yes": old_no_new_yes,
                "return_now_yes_option_no": now_yes_action_no,
                "return_now_no_option_yes": now_no_action_yes,
                "mixed_option_anchors": mixed,
                "mixed_option_anchor_fraction": float(mixed / len(by_anchor)),
            }
        )
    return reports
