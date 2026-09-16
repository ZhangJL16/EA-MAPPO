"""Complete-option data and on-policy score loss; no collector or training run.

The frozen navigation reward is never changed. Task counts and first-event
labels below define a separate high-level research objective. A skill deadline
is operational failure, NOT an observed battery depletion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

FEATURE_DIM = 2115  # Existing FrozenOptionFeatures, without evaluation countdown.
TERMINALS = frozenset({'evaluation_time', 'energy_exhausted', 'task_step_limit',
                      'return_deadline', 'episode_emergency_step_guard'})
OPERATIONAL_FAILURES = TERMINALS - {'evaluation_time'}


@dataclass(frozen=True)
class OptionRecord:
    features: np.ndarray
    action: int  # 0: continue, 1: commit and execute return until service/end
    behavior_log_prob: float
    trainable: bool  # False for a forced continue at the station.
    start_step: int
    end_step: int
    start_seconds: float
    end_seconds: float
    task_gain: int
    collision_count: int
    recharged: bool
    end_reason: str | None


def pack_option(features: np.ndarray, action: int, behavior_log_prob: float,
                trainable: bool, start_step: int, start_seconds: float,
                infos: Sequence[dict]) -> OptionRecord:
    """Consume EVERY physics-policy step in an option, not E1's sparse trace.

    infos are unmodified PersistentRecharge.step info dictionaries. The caller
    owns option boundaries, behavior policy identity, and legal feature encoding.
    No simulator geometry or worker/job metadata is accepted as a feature field.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.shape != (FEATURE_DIM,) or not np.isfinite(x).all():
        raise ValueError('finite frozen sensor features required')
    if (action not in (0, 1) or type(trainable) is not bool
            or not np.isfinite(behavior_log_prob) or behavior_log_prob > 0
            or (not trainable and (action != 0 or behavior_log_prob != 0))):
        raise ValueError('valid sampled action/log probability or forced continue required')
    if (not infos or not isinstance(start_step, int) or start_step < 0
            or not np.isfinite(start_seconds) or start_seconds < 0):
        raise ValueError('nonempty complete option and valid start required')
    previous_time = float(start_seconds)
    tasks = contacts = 0
    reason = None
    for offset, info in enumerate(infos, start=1):
        if info['step'] != start_step+offset:
            raise ValueError('missing or duplicated physical-policy step')
        now = float(info['simulation_seconds'])
        if not np.isfinite(now) or now <= previous_time:
            raise ValueError('physical clock must advance')
        gain, cost = info['task_gain'], info['cost']
        if (not np.isfinite(gain) or gain < 0 or int(gain) != gain
                or cost not in (0, 1)):
            raise ValueError('integer tasks and unified binary contact cost required')
        row = info.get('persistent_run')
        ended = row is not None
        if (info['recharge_event'] or ended) and offset != len(infos):
            raise ValueError('option crosses a service or run boundary')
        if ended:
            reason = row['termination']
            if reason not in TERMINALS:
                raise ValueError('unknown terminal semantics')
        tasks += int(gain)
        contacts += int(cost)
        previous_time = now
    recharged = bool(infos[-1]['recharge_event'])
    if (action == 1 and not recharged and reason is None) or (action == 0 and recharged):
        raise ValueError('return must run to service/end; continue cannot recharge')
    return OptionRecord(x.copy(), action, float(behavior_log_prob), trainable,
                        start_step, start_step+len(infos), float(start_seconds),
                        previous_time, tasks, contacts, recharged, reason)


@dataclass(frozen=True)
class CompleteRun:
    world_id: int
    policy_id: str  # Full frozen collection policy/checkpoint identity.
    evaluation_seconds: float
    options: tuple[OptionRecord, ...]


def run_returns(run: CompleteRun) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return-to-go for tasks/hour, operational failure, and first contact.

    Stops only at the predeclared run endpoint, never at a recharge. A pause or
    unfinished return is rejected instead of fabricated as zero-risk terminal.
    Operational failure includes administrative skill/guard deadlines; retain
    end_reason to separately report depletion and censoring in evaluation.
    """
    if (not run.policy_id or not run.options or not np.isfinite(run.evaluation_seconds)
            or run.evaluation_seconds <= 0):
        raise ValueError('complete identified run and positive horizon required')
    tasks, failures, contacts = [], [], []
    prior_step, prior_seconds, had_contact = 0, 0., False
    for i, option in enumerate(run.options):
        if (option.start_step != prior_step
                or not np.isclose(option.start_seconds, prior_seconds, rtol=0, atol=1e-8)
                or option.end_step <= option.start_step
                or not np.isfinite(option.end_seconds)
                or option.end_seconds <= option.start_seconds
                or option.task_gain < 0 or int(option.task_gain) != option.task_gain
                or option.collision_count < 0 or int(option.collision_count) != option.collision_count
                or option.collision_count > option.end_step-option.start_step):
            raise ValueError('complete contiguous run with valid counts required')
        last = i == len(run.options)-1
        if (last and option.end_reason not in TERMINALS) or (not last and option.end_reason is not None):
            raise ValueError('unfinished or internally terminated run')
        if option.start_seconds >= run.evaluation_seconds+1e-8:
            raise ValueError('option starts beyond evaluation horizon')
        tasks.append(option.task_gain*3600/run.evaluation_seconds)
        failures.append(float(option.end_reason in OPERATIONAL_FAILURES))
        contacts.append(float(option.collision_count > 0 and not had_contact))
        had_contact |= option.collision_count > 0
        prior_step, prior_seconds = option.end_step, option.end_seconds
    if run.options[-1].end_reason == 'evaluation_time' and prior_seconds+1e-8 < run.evaluation_seconds:
        raise ValueError('evaluation horizon has not been reached')
    return tuple(np.cumsum(np.asarray(v, dtype=np.float64)[::-1])[::-1].copy()
                 for v in (tasks, failures, contacts))


def score_loss(log_probs: Sequence[torch.Tensor], runs: Sequence[CompleteRun],
               policy_id: str, lambda_operational: float, lambda_contact: float,
               baselines: Sequence[torch.Tensor] | None = None,
               weights: Sequence[float] | None = None) -> torch.Tensor:
    """One on-policy Monte Carlo score-gradient step, averaged over RUNS.

    Not PPO, CPO, a multi-epoch replay loss, or a safety certificate. Baselines
    must be action-independent and fitted independently of the current outcomes
    (e.g. a frozen preceding-batch baseline). They are detached here. Optional
    fixed weights express an explicitly declared initial-world distribution.
    The caller must bind policy_id to the actual collection checkpoint; checking
    sampled log probabilities alone cannot establish full policy equivalence.
    """
    if (not runs or len(log_probs) != len(runs)
            or (baselines is not None and len(baselines) != len(runs))
            or any(not np.isfinite(x) or x < 0 for x in (lambda_operational, lambda_contact))):
        raise ValueError('aligned nonempty on-policy runs and nonnegative multipliers required')
    w = np.ones(len(runs)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(runs),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('fixed nonnegative run weights required')
    losses = []
    for i, (lp, run) in enumerate(zip(log_probs, runs)):
        if run.policy_id != policy_id:
            raise ValueError('collection policy mismatch: recollect after an update')
        utility, operational, contact = run_returns(run)
        if lp.shape != (len(run.options),) or not torch.isfinite(lp).all():
            raise ValueError('one finite log probability per option required')
        old = lp.new_tensor([o.behavior_log_prob for o in run.options])
        if not torch.allclose(lp.detach(), old, atol=1e-6, rtol=1e-6):
            raise ValueError('stale behavior probabilities: not on-policy')
        target = lp.new_tensor(utility-lambda_operational*operational-lambda_contact*contact)
        baseline = torch.zeros_like(lp) if baselines is None else baselines[i]
        if baseline.shape != lp.shape or not torch.isfinite(baseline).all():
            raise ValueError('finite aligned baseline required')
        mask = lp.new_tensor([o.trainable for o in run.options])
        losses.append(-(lp*(target-baseline.detach()).detach()*mask).sum()*float(w[i]/w.sum()))
    return torch.stack(losses).sum()
