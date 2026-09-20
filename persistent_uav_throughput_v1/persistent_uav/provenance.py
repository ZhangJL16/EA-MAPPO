import hashlib
import importlib.metadata
from pathlib import Path

from .navigation import LEGACY, MODEL, MODEL_SHA
from .storage import sha

ROOT = Path(__file__).resolve().parents[1]


def provenance():
    if sha(MODEL) != MODEL_SHA:
        raise RuntimeError('frozen checkpoint changed')
    files = list((ROOT / 'persistent_uav').glob('*.py'))
    files += list((LEGACY / 'experiments/directional_navigation').glob('*.py'))
    files += [LEGACY / 'envs/UAVEnergyDelivery.py', LEGACY / 'envs/UAVEnergyDeliverySAC.py']
    files += list((LEGACY / 'runtime_support/review_bundle').rglob('*.py'))
    return dict(checkpoint=str(MODEL), checkpoint_sha256=MODEL_SHA,
                sources={str(p): sha(p) for p in sorted(set(files))},
                packages={p: importlib.metadata.version(p) for p in
                          ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'scipy')},
                training_updates=0, optimizer_loaded=False)


def verify_provenance(expected):
    if provenance() != expected:
        raise RuntimeError('source/model/dependency contract changed; resume refused')

