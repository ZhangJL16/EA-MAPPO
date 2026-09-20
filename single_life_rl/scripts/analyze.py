"""Predeclared post-completion analysis; no feedback into experiments."""
import collections
import csv
import json
import math
from pathlib import Path
import numpy as np
from single_life_rl.scripts.io_utils import write_json


def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    middle=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,middle-radius),min(1.,middle+radius)]


def summarize(rows):
    n=len(rows); failures=sum(r['catastrophe'] for r in rows)
    times=[r['certification_time'] for r in rows if r['certification_time'] is not None]
    return dict(runs=n,catastrophes=failures,catastrophe_rate=failures/n,catastrophe_wilson95=wilson(failures,n),
                mean_regret=float(np.mean([r['regret'] for r in rows])),
                certified=len(times),certification_fraction=len(times)/n,
                mean_certified_time=None if not times else float(np.mean(times)),
                restricted_mean_certification_time=float(np.mean([r['certification_time'] if r['certification_time'] is not None else r['spec']['horizon'] for r in rows])),
                mean_exploration_time=float(np.mean([r['exploration_time'] for r in rows])),
                true_model_eliminations=sum(r['true_model_eliminated'] for r in rows),
                mean_unlocked=float(np.mean([r['unlocked_experiments'] for r in rows])),
                mean_cycles=float(np.mean([r['cycles'] for r in rows])))


def analyze(output):
    output=Path(output);aggregate=output/'aggregate'
    integrity=json.loads((aggregate/'integrity.json').read_text());assert integrity['passed']
    allrows=json.loads((aggregate/'results.json').read_text())
    available=[r for r in allrows if 'catastrophe' in r]
    primary=[r for r in available if not r['spec'].get('sensitivity',False)]
    groups=collections.defaultdict(list)
    for r in primary:groups[(r['suite'],r['method'])].append(r)
    summary=[dict(suite=k[0],method=k[1],**summarize(v)) for k,v in sorted(groups.items())]
    write_json(aggregate/'summary.json',summary)
    # Stratify fixed models/parameter cells; pooled intervals only descriptive.
    cells=collections.defaultdict(list)
    for r in available:
        s=r['spec'];condition={k:s[k] for k in ('kappa','kappas','m','truth','truth_index','delta') if k in s}
        if 'tree' in s:condition['tree_id']=s['tree']['tree_id']
        cells[(r['suite'],r['method'],json.dumps(condition,sort_keys=True))].append(r)
    write_json(aggregate/'cells.json',[dict(suite=k[0],method=k[1],condition=json.loads(k[2]),**summarize(v)) for k,v in sorted(cells.items())])
    with (aggregate/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    binary=[r for r in primary if r['suite']=='binary' and r['method']=='SafeRefine' and r['spec']['kappa']>0]
    pts=[]
    for k in sorted({r['spec']['kappa'] for r in binary}):
        rr=[r for r in binary if r['spec']['kappa']==k];s=summarize(rr)
        pts.append((rr[0]['sum_inverse_stage_information'],s['restricted_mean_certification_time']))
        axes[0,0].annotate(f'k={k}; {s["certification_fraction"]:.0%} certified',pts[-1],fontsize=7)
    axes[0,0].plot(*zip(*pts),'o-');axes[0,0].set(xlabel='1 / Gamma (binary)',ylabel='Restricted mean certification time')
    for k in (.02,.04,.08,.16):
        pts=[]
        for d in (.1,.05,.01,.001):
            rr=[r for r in available if r['suite']=='binary' and r['method']=='SafeRefine' and r['spec']['kappa']==k and r['spec']['delta']==d]
            pts.append((math.log(1/d),summarize(rr)['restricted_mean_certification_time']))
        axes[0,1].plot(*zip(*pts),'o-',label=f'k={k}')
    axes[0,1].set(xlabel='log(1 / lifetime delta)',ylabel='Restricted mean certification time');axes[0,1].legend()
    for method in ('SafeRefine','FullModelID'):
        pts=[]
        for m in (0,4,8,16,32):
            rr=[r for r in primary if r['suite']=='nuisance' and r['method']==method and r['spec']['m']==m]
            pts.append((m,summarize(rr)['mean_exploration_time']))
        axes[1,0].plot(*zip(*pts),'o-',label=method)
    axes[1,0].set(xlabel='Nuisance dimensions',ylabel='Mean exploration time');axes[1,0].legend()
    pts=[]
    for tid in range(100):
        rr=[r for r in primary if r['suite']=='random' and r['method']=='SafeRefine' and r['spec']['tree']['tree_id']==tid]
        pts.append((float(np.mean([r['sum_inverse_stage_information'] for r in rr])),summarize(rr)['restricted_mean_certification_time']))
    axes[1,1].scatter(*zip(*pts),s=12);axes[1,1].set(xlabel='Mean sum of inverse stage information',ylabel='Restricted mean certification time')
    fig.savefig(aggregate/'scaling.png',dpi=160);fig.savefig(aggregate/'scaling.pdf');plt.close(fig)
    lines=['# Frozen single-life protocol v1 results','',f'Integrity: {integrity["unique"]}/{integrity["expected"]} unique jobs, no missing/duplicates.',
           '', 'Primary delta = 0.05. Lifetime safety is a theorem under the stated model assumptions; finite-horizon frequencies and Wilson intervals are calibration checks, not proofs. Pooled heterogeneous-cell intervals are descriptive. Uncertified lives enter restricted means at their horizon, including catastrophe.','',
           '| Suite | Method | Runs | Catastrophe | Certified | Mean regret | Mean exploration |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["suite"]} | {r["method"]} | {r["runs"]} | {r["catastrophes"]} | {r["certified"]} | {r["mean_regret"]:.2f} | {r["mean_exploration_time"]:.2f} |')
    lines+=['','No result triggered parameter changes or a follow-on experiment. See cells.json for stratification and results.json for every life, including failures.',
            '', 'UAV observes the full deterministic cycle clock; support filtering can identify theta after one completed safe mission. Gaussian-only KL scaling is not claimed. No task distribution or battery adjustment is permitted if optimal arms coincide.',
            '', 'The matching information bound is established only for the staged Bernoulli witness family in the theory document, not for arbitrary unknown MDPs.','']
    (aggregate/'REPORT.md').write_text('\n'.join(lines))
