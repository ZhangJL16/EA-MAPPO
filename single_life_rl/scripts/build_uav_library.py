"""200 fixed physical missions, resumable at real low-level policy steps."""
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import numpy as np
from single_life_rl.scripts.io_utils import write_json, save_pickle, digest


def mission_worker(args):
    output, manifest_path, worker, workers = args
    root = Path(output); folder = root/f'worker_{worker:02d}'; folder.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (folder/'process.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    repo = Path(__file__).resolve().parents[2]
    os.environ['PERSISTENT_UAV_LEGACY_ROOT'] = str(repo)
    sys.path.insert(0,str(repo/'persistent_uav_throughput_v1'))
    from persistent_uav.navigation import FrozenNavigator
    from persistent_uav.storage import snapshot, restore
    manifest = json.loads(Path(manifest_path).read_text())
    assigned = manifest['missions'][worker::workers]
    checkpoint = folder/'resume.json'
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0,True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0,True))
    if checkpoint.exists():
        state = restore(folder)
        if state['manifest_sha256'] != digest(manifest_path): raise RuntimeError('manifest mismatch')
    else:
        state = dict(rows=[], nav=None, phase='outbound', energy=0., steps=0, manifest_sha256=digest(manifest_path))
    def save(status):
        h = snapshot(folder, state)
        write_json(folder/'status.json', dict(status=status, completed=len(state['rows']), assigned=len(assigned),
                   steps=state['steps'], snapshot=h, checkpoint_unix=time.time(), training_updates=0))
    save('running')
    try:
        while len(state['rows']) < len(assigned):
            job = assigned[len(state['rows'])]
            if state['nav'] is None:
                state['nav'] = FrozenNavigator(manifest['pilot_battery'], layout=manifest['layout'], option_step_limit=manifest['option_step_limit'])
                state['nav'].start_leg(job['goal']); state['phase']='outbound'; state['energy']=0.
            nav=state['nav']
            reached=nav.reached(); timeout=False
            if not reached:
                result=nav.advance_flight(.2); state['steps']+=1; state['energy']+=result.energy_used
                reached=result.reached; timeout=result.timeout
                if result.depleted: raise RuntimeError('pilot battery exhausted')
            if reached and state['phase']=='outbound':
                nav.stop_at_service(); nav.start_leg(nav.station); state['phase']='return'
            elif reached or timeout:
                state['rows'].append(dict(mission_id=job['mission_id'], goal=job['goal'], success=bool(reached),
                     timeout=bool(timeout and not reached), failure_phase=None if reached else state['phase'],
                     energy=state['energy'], duration=nav.time, collision_count=nav.contacts, policy_steps=nav.policy_steps))
                nav.close(); state['nav']=None; save('running')
            if state['steps'] % 100 == 0 or stopped[0]:
                save('paused' if stopped[0] else 'running')
                if stopped[0]: return {'worker':worker,'paused':True}
        write_json(folder/'results.json',state['rows']); save('complete')
        return {'worker':worker,'complete':True}
    except Exception as e:
        write_json(folder/'error.json',dict(error=repr(e), resume='last complete checkpoint only'))
        raise


def prepare_library(config, output):
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    path=root/'manifest.json'
    if path.exists(): return path
    repo=Path(__file__).resolve().parents[2]
    os.environ['PERSISTENT_UAV_LEGACY_ROOT']=str(repo)
    sys.path.insert(0,str(repo/'persistent_uav_throughput_v1'))
    from persistent_uav.streams import legal_position
    frozen=json.loads((repo/'persistent_uav_throughput_v1/evidence/v1_2/calibration/frozen_regimes.json').read_text())
    rng=np.random.default_rng(config['uav']['library_seed'])
    missions=[dict(mission_id=i,goal=legal_position(rng,frozen['layout'])) for i in range(config['uav']['missions'])]
    write_json(path,dict(missions=missions, layout=frozen['layout'], seed=config['uav']['library_seed'],
               pilot_battery=config['uav']['pilot_battery'],option_step_limit=4000,
               sampling='200 service locations fixed before physical outcomes; no replacement of timeout missions'))
    return path


def collect_library(config, output):
    root=Path(output); manifest=json.loads((root/'manifest.json').read_text()); rows=[]
    for folder in sorted(root.glob('worker_*')):
        status=json.loads((folder/'status.json').read_text())
        if status['status']!='complete': raise RuntimeError('incomplete library worker')
        rows.extend(json.loads((folder/'results.json').read_text()))
    rows.sort(key=lambda r:r['mission_id'])
    assert [r['mission_id'] for r in rows]==list(range(config['uav']['missions']))
    successes=[r for r in rows if r['success']]
    B=float(np.quantile([r['energy'] for r in successes], config['uav']['battery_quantile'],method='linear')) if successes else None
    library=dict(missions=rows, successes=len(successes), timeouts=sum(r['timeout'] for r in rows),
                 battery=B, recharge_rate=B/config['uav']['calibration_median_time'] if B is not None else None,
                 battery_rule='successful-mission energy quantile, linear interpolation; no algorithm outcomes used',
                 manifest_sha256=digest(root/'manifest.json'))
    write_json(root/'library.json',library)
    return library
