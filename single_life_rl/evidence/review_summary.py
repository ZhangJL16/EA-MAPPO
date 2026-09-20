"""Read-only interpretation audit after every frozen job completed."""
import json
from pathlib import Path
import sys
import numpy as np
from single_life_rl.scripts.analyze import summarize
from single_life_rl.envs.uav_mission_library import MissionModels

root=Path(sys.argv[1]);out=Path(__file__).resolve().parent/'complete_v1'
assert json.loads((root/'status.json').read_text())['status']=='complete'
rows=json.loads((root/'aggregate/results.json').read_text())
primary=[r for r in rows if not r['spec'].get('sensitivity',False)]
safe=[r for r in rows if r['method']=='SafeRefine']
assert all(not r['catastrophe'] or r['true_model_eliminated'] for r in safe)
summary={}
for suite in ('binary','recursive','nuisance','random'):
    rr=[r for r in primary if r['suite']==suite and r['method']=='SafeRefine']
    summary[suite]=dict(**summarize(rr),correct_certificates=sum(r['correct_certificate'] is True for r in rr),
           incorrect_certificates=sum(r['correct_certificate'] is False for r in rr))
binary=[]
for k in (0,.02,.04,.08,.16):
    rr=[r for r in primary if r['suite']=='binary' and r['method']=='SafeRefine' and r['spec']['kappa']==k]
    binary.append(dict(kappa=k,inverse_information=rr[0]['sum_inverse_stage_information'],**summarize(rr)))
sensitivity=[]
for d in (.1,.05,.01,.001):
    rr=[r for r in rows if r['suite']=='binary' and r['method']=='SafeRefine' and r['spec']['delta']==d and r['spec']['kappa']>0]
    sensitivity.append(dict(delta=d,**summarize(rr)))
nuisance=[]
for m in (0,4,8,16,32):
    nuisance.append(dict(m=m,**{method:summarize([r for r in primary if r['suite']=='nuisance' and r['method']==method and r['spec']['m']==m]) for method in ('SafeRefine','FullModelID')}))
lib=json.loads((root/'library/library.json').read_text());config=json.loads((root/'freeze.json').read_text())['config']
models=MissionModels(lib,config['uav']['theta'])
uav=[dict(theta=t,safe_arms=len(models.safe_experiments([i])),optimal_mission_id=models.optimal(i)['mission_id']) for i,t in enumerate(models.thetas)]
result=dict(primary_safe_refine=summary,binary_by_kappa=binary,delta_sensitivity_positive_kappa=sensitivity,
            nuisance=nuisance,uav=uav,catastrophe_implies_true_model_excluded_check=True,
            statement='All analyses after completion. Issued certificate is not necessarily correct; rows retain failures.')
(out/'review_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'safe_refine':summary,'uav':uav}))
