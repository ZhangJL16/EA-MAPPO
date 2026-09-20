import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np


def smoke(output):
    """Tiny real frozen-actor execution and a mid-flight disk-resume equivalence check."""
    from .baselines import Scheduler
    from .config import Config
    from .environment import PersistentUAVThroughput
    from .estimates import EstimateModel
    from .navigation import FrozenNavigator
    from .storage import restore, snapshot, write_json
    from .streams import workload

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('smoke output must be empty')
    config = Config(capacity=100., recharge_rate=1., arrival_rate=.01, cutoff=1.)
    nav = FrozenNavigator(config.capacity)
    env = PersistentUAVThroughput(config, nav, workload(120260920, config, nav.layout))
    scheduler = Scheduler('fifo', EstimateModel((0., .1, .2), (0., .01, .02)))
    env.step(scheduler.choose(env.observe()))
    snapshot(output, env)
    resumed = restore(output)
    env.step()
    resumed.step()
    if env.observe() != resumed.observe() or env.summary() != resumed.summary():
        raise AssertionError('actual frozen-actor mid-flight resume did not reproduce next state')
    if not np.isfinite(env.nav.observation()).all() or env.nav.policy_steps != 2:
        raise AssertionError('nonfinite or unexpected frozen execution')
    write_json(output / 'smoke.json', dict(
        kind='ENGINEERING_SMOKE_ONLY_NOT_CALIBRATION_OR_BASELINE_RESULT',
        passed=True, config=asdict(config), steps_per_branch=2,
        midflight_resume_next_state_equal=True, physical_resets=env.nav.reset_count,
        observation_dim=int(env.nav.observation_space.shape[0]), training_updates=0,
        collision_count=env.nav.contacts))
    nav.close()
    resumed.nav.close()
    print('frozen-actor smoke and mid-flight disk resume passed', flush=True)


def main():
    p = argparse.ArgumentParser(description='PersistentUAVThroughput-v1; no neural training commands')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('smoke')
    s.add_argument('--output', type=Path, required=True)
    for name in ('calibrate', 'baselines'):
        a = sub.add_parser(name)
        a.add_argument('--output', type=Path, required=True)
        a.add_argument('--resume', action='store_true')
        a.add_argument('--checkpoint-policy-steps', type=int, default=100)
        a.add_argument('--stop-after-checkpoint', action='store_true')
        if name == 'baselines':
            a.add_argument('--frozen', type=Path, required=True)
            a.add_argument('--split', choices=('validation', 'evaluation'), required=True)
            a.add_argument('--regimes', required=True, help='all or comma-separated IDs 0..26')
            a.add_argument('--methods', help='comma-separated baseline names; default all on evaluation')
            a.add_argument('--threshold-file', type=Path)
            a.add_argument('--calibration-compatibility', type=Path,
                           help='audited source change record; does not rewrite frozen calibration')
    args = p.parse_args()
    if args.command == 'smoke':
        smoke(args.output)
    elif args.command == 'calibrate':
        from .calibration import run
        from .storage import exclusive_run
        with exclusive_run(args.output):
            run(args.output, resume=args.resume, checkpoint_policy_steps=args.checkpoint_policy_steps,
                stop_after_checkpoint=args.stop_after_checkpoint)
    else:
        from .evaluation import run
        ids = list(range(27)) if args.regimes == 'all' else [int(i) for i in args.regimes.split(',')]
        from .storage import exclusive_run
        with exclusive_run(args.output):
            run(args.frozen, args.output, split=args.split, regime_ids=ids,
                methods=None if not args.methods else args.methods.split(','),
                threshold_file=args.threshold_file, resume=args.resume,
                compatibility_file=args.calibration_compatibility,
                checkpoint_policy_steps=args.checkpoint_policy_steps,
                stop_after_checkpoint=args.stop_after_checkpoint)


if __name__ == '__main__':
    main()
