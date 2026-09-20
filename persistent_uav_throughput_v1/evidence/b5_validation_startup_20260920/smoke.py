"""One tiny real-navigation B5 telemetry smoke; not a diagnostic result."""
import os,sys,json,copy
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];PROJECT=BASE/'repo/persistent_uav_throughput_v1'
os.environ['PERSISTENT_UAV_LEGACY_ROOT']=str(BASE/'repo');sys.path.insert(0,str(PROJECT))
from persistent_uav.baselines import Scheduler
from persistent_uav.config import Config
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.estimates import EstimateModel
from persistent_uav.navigation import FrozenNavigator
from persistent_uav.streams import workload
from persistent_uav.storage import snapshot,restore,write_json
from persistent_uav.diagnostics import audited_step,run_diagnostics
import numpy as np
OUT=Path(__file__).resolve().parent/'smoke';OUT.mkdir(exist_ok=False)
frozen=json.loads((PROJECT/'evidence/v1_2/calibration/frozen_regimes.json').read_text())
config=Config(capacity=100.,recharge_rate=1.,arrival_rate=.01,cutoff=1.)
nav=FrozenNavigator(config.capacity,layout=frozen['layout'])
env=PersistentUAVThroughput(config,nav,workload(120260920,config,frozen['layout']))
model=EstimateModel(**frozen['model']);scheduler=Scheduler('reserve_sjf',model)
audited_step(env,scheduler.choose(env.observe()),model)
snapshot(OUT/'midflight',env);recovered=restore(OUT/'midflight')
for branch in (env,recovered):
 while not branch.done:
  action=scheduler.choose(branch.observe()) if branch.decision_required else None
  audited_step(branch,action,model)
 assert np.isfinite(branch.nav.observation()).all()
assert env.summary()==recovered.summary() and env.events==recovered.events
flags,decisions=run_diagnostics(env.events,env.summary());assert decisions and decisions[0]['candidates']
assert all(d['margin_added']==0. for d in decisions)
write_json(OUT/'smoke.json',dict(kind='ENGINEERING_ONLY_NOT_VALIDATION',passed=True,midflight_recovery_equal=True,finite_observation=True,training_updates=0,summary=env.summary(),decision_diagnostics=decisions))
nav.close();recovered.nav.close();print('Real B5 telemetry smoke and midflight recovery passed.')
