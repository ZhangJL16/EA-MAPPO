"""Scripted integration evidence, NOT a scheduler benchmark or calibration sample."""
from dataclasses import asdict
from pathlib import Path
import json

from .baselines import Action
from .config import Config
from .environment import PersistentUAVThroughput
from .navigation import FrozenNavigator
from .storage import restore, snapshot, write_json
from .streams import Task


def run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('smoke output must be empty')
    c = Config(100., 2., .01, 500.)
    nav = FrozenNavigator(c.capacity)
    home = nav.station
    targets = [home - [100, 0, 0], home + [-100, 100, 0], home - [100, 0, 0]]
    tasks = tuple(Task(i, 0., 0., tuple(p.tolist())) for i, p in enumerate(targets))
    env = PersistentUAVThroughput(c, nav, tasks)
    actions = [Action('serve', 0, 'scripted_smoke'), Action('serve', 1, 'scripted_smoke'),
               Action('recharge', reason='scripted_smoke'), Action('serve', 2, 'scripted_smoke')]
    env.step(actions[0])
    env.step()
    snapshot_time = env.time
    snapshot(output / 'midflight', env)
    resumed = restore(output / 'midflight')

    def finish(instance):
        action_index = 1
        while instance.completed < 3:
            if instance.done:
                raise AssertionError(f'continuing smoke ended early: {instance.summary()}')
            if instance.mode == 'IDLE':
                if action_index >= len(actions):
                    raise AssertionError('unexpected extra action required')
                instance.step(actions[action_index])
                action_index += 1
            else:
                instance.step()
        assert instance.completed_recharges == 1 and instance.nav.reset_count == 1
        assert instance.time_by_mode['charging'] > 0 and instance.time_by_mode['flight'] > 0
        return dict(summary=instance.summary(), final_observation=instance.observe(), events=instance.events)

    uninterrupted = finish(env)
    recovered = finish(resumed)
    if uninterrupted != recovered:
        raise AssertionError('full continuing outcome differs after midflight disk resume')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'continuing_trace.json', dict(
        kind='ENGINEERING_FIXTURE_NOT_BASELINE_RESULT', config=asdict(c),
        scripted_actions=[asdict(a) for a in actions], **uninterrupted))
    write_json(output / 'recovery_equivalence.json', dict(
        passed=True, comparison='full remaining serve->serve->recharge->serve event log, summary and final observation',
        snapshot_time=snapshot_time, restored_outcome_equal=True, training_updates=0,
        physical_resets=env.nav.reset_count, completed_services=env.completed,
        completed_recharges=env.completed_recharges))
    nav.close()
    resumed.nav.close()
    print(json.dumps(dict(smoke_passed=True, completed_services=3, full_outcome_resume_equal=True)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
