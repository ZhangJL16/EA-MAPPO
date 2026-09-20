"""Offline privileged navigation diagnostics. Never imported by a scheduler."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .evaluation import wilson_interval
from .storage import write_json


def exposure(start, goal, layout):
    a, b = np.asarray(start)[:2], np.asarray(goal)[:2]
    d = b - a
    count = 0
    for obstacle in layout:
        center = np.asarray(obstacle['position'])
        projection = np.clip(np.dot(center - a, d) / max(np.dot(d, d), 1e-12), 0., 1.)
        count += np.linalg.norm(center - (a + projection * d)) <= obstacle['radius'] + 5.5
    return int(count)


def cell(rows):
    n = len(rows)
    success = sum(r['success'] for r in rows)
    failed = n - success
    reasons = Counter('navigation_timeout' if r['timeout'] else 'other_failure'
                      for r in rows if not r['success'])
    return dict(n=n, success=success, failures=failed,
                success_rate=success / n if n else None,
                success_wilson95=wilson_interval(success, n),
                failure_wilson95=wilson_interval(failed, n), failure_reasons=dict(reasons))


def analyze(folder, output):
    folder, output = Path(folder), Path(output)
    rows = json.loads((folder / 'jobs.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    stats = json.loads((folder / 'statistics.json').read_text())
    annotated = []
    for row in rows:
        d = np.asarray(row['goal']) - row['start']
        annotated.append(dict(**row, horizontal_distance=float(np.linalg.norm(d[:2])),
                              vertical_distance=float(abs(d[2])),
                              privileged_chord_exposure=exposure(row['start'], row['goal'], manifest['layout'])))
    groups = {}
    for key, cuts in [('horizontal_distance', (1000, 2000, 3000)),
                      ('vertical_distance', (60, 180)), ('privileged_chord_exposure', (1, 2))]:
        bins = []
        bounds = (0,) + cuts + (float('inf'),)
        for lo, hi in zip(bounds, bounds[1:]):
            subset = [r for r in annotated if lo <= r[key] < hi]
            bins.append(dict(lower=lo, upper=None if np.isinf(hi) else hi, **cell(subset)))
        groups[key] = bins
    overall = cell(rows)
    failed_time = sum(r['duration'] for r in rows if not r['success'])
    total_time = sum(r['duration'] for r in rows)
    fraction = failed_time / total_time if total_time else 0.
    signals = []
    if overall['failure_wilson95'] and overall['failure_wilson95'][0] > .05:
        signals.append('overall_failure_wilson_lower_gt_5_percent')
    for group, bins in groups.items():
        for b in bins:
            if b['n'] >= 30 and b['failure_wilson95'][0] > .1:
                signals.append(f"{group}_[{b['lower']},{b['upper']})_failure_lower_gt_10_percent")
    if fraction >= .1:
        signals.append('failed_jobs_consumed_at_least_10_percent_of_pilot_time')
    status = ('INCOMPLETE' if len(rows) != 1000 else 'PAUSE_SCHEDULING' if signals else 'REVIEW_REQUIRED')
    result = dict(status=status, overall=overall, statistics=stats, strata=groups,
                  failed_job_time_fraction=fraction, stopping_signals=signals,
                  hocbf_fallback_steps=sum(r['hocbf_fallback_steps'] for r in rows),
                  timeout_after_any_contact=sum(r['timeout'] and r['collision_count'] > 0 for r in rows),
                  timeout_without_contact=sum(r['timeout'] and r['collision_count'] == 0 for r in rows),
                  training_updates=0, baseline_runs=0,
                  caveat='Pre-analysis of already-started pilot, not prospective preregistration. Ranking dominance untested.')
    write_json(output / 'qualification.json', result)
    write_json(output / 'annotated_jobs.json', annotated)
    lines = ['# Navigation qualification', '', f'Status: **{status}**; baseline runs = 0; neural updates = 0.', '',
             f"Success: {overall['success']}/{overall['n']} ({overall['success_rate']:.3%}); Wilson 95% CI: {overall['success_wilson95']}.", '',
             'Quantiles below condition on navigation success; failed routes remain in the evidence.', '',
             '| Metric | q25 | q50 | q75 | q90 |', '|---|---:|---:|---:|---:|']
    for metric in ('time', 'energy'):
        if metric in stats:
            q = stats[metric]
            lines.append(f"| {metric} | {q['q25']:.6f} | {q['median']:.6f} | {q['q75']:.6f} | {q['q90']:.6f} |")
    lines += ['', f'Failure reasons: {overall["failure_reasons"]}.',
              f'Pilot time consumed by failed jobs: {fraction:.3%}; this is not an additive scheduling loss estimate.', '',
              '| Diagnostic | Range | n | Failures | Success rate |', '|---|---|---:|---:|---:|']
    for key, bins in groups.items():
        for b in bins:
            rate = f"{b['success_rate']:.2%}" if b['n'] else 'N/A'
            lines.append(f"| {key} | [{b['lower']}, {b['upper'] or 'inf'}) | {b['n']} | {b['failures']} | {rate} |")
    lines += ['', 'Stopping signals: ' + (', '.join(signals) or 'none; user review remains required'), '',
              'Obstacle exposure is a privileged straight-chord geometry diagnostic, not actor input or realized flight exposure.',
              'Calibration cannot prove navigation failure does not dominate future policy rankings. No baseline was run.']
    (output / 'NAVIGATION_QUALIFICATION.md').write_text('\n'.join(lines) + '\n')
    return result
