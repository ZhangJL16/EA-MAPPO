"""Bounded engineering replay of the saved run-383 failure; never a diagnostic run."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persistent_uav import navigation
from persistent_uav.baselines import Scheduler
from persistent_uav.estimates import EstimateModel
from persistent_uav.storage import restore, write_json

out = ROOT / 'evidence/v1_4'
old_source = subprocess.check_output(['git', '-C', str(ROOT.parent), 'show',
    'b8cb673:persistent_uav_throughput_v1/persistent_uav/navigation.py'], text=True)
old_class = next(n for n in ast.parse(old_source).body if isinstance(n, ast.ClassDef) and n.name == 'FrozenNavigator')
method = next(n for n in old_class.body if isinstance(n, ast.FunctionDef) and n.name == 'advance_stationary')
ns = dict(vars(navigation))
exec(compile(ast.Module(body=[method], type_ignores=[]), '<historical stationary method>', 'exec'), ns)
left, right = [restore(out / 'original_failure') for _ in range(2)]
old, fixed = left['env'], right['env']
old.nav.advance_stationary = types.MethodType(ns['advance_stationary'], old.nav)
job = left['contract']['jobs'][len(left['rows'])]
frozen = json.loads((ROOT / 'evidence/v1_2/calibration/frozen_regimes.json').read_text())
scheduler = Scheduler(job['method'], EstimateModel(**frozen['model']), job['threshold'])
for step in range(101):
    assert old.observe() == fixed.observe() and old.summary() == fixed.summary()
    assert old.events == fixed.events
    action = scheduler.choose(old.observe()) if old.decision_required else None
    before = fixed.observe()
    try:
        old.step(action)
    except ValueError as error:
        assert str(error) == 'stationary duration must lie on the physics grid'
        next_arrival = fixed._tasks[fixed.cursor].arrival
        fixed.step(action)
        fixed._assert_invariants()
        assert abs(fixed.time - next_arrival) < 1e-8
        assert fixed.decision_required and fixed.queue and fixed.failure is None
        assert fixed.nav.reset_count == 1
        write_json(out / 'wait_grid_replay.json', dict(
            kind='BOUNDED_ENGINEERING_REPLAY_NOT_BASELINE_RESULT', passed=True,
            job=job, steps_from_checkpoint=step+1, historical_error=str(error),
            all_preceding_observations_events_summaries_identical=True,
            before=before, after=fixed.observe(), snapshot_completed_rows=len(left['rows']),
            physical_resets=1, neural_updates=0, production_run_resumed=False))
        print(json.dumps(dict(passed=True, step=step+1, before_time=before['time'],
                              after_time=fixed.time, mode=fixed.mode, battery=fixed.nav.energy)))
        break
    else:
        fixed.step(action)
else:
    raise AssertionError('saved failure did not occur within one checkpoint interval')
old.nav.close()
fixed.nav.close()
