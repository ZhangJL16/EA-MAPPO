"""Algebra on existing P2-A2 receipts ONLY. No simulator/actor imports or sampling."""
import argparse
import hashlib
import json
from pathlib import Path


def hashed(path):
    v=json.loads(path.read_text())
    h=hashlib.sha256(json.dumps(v['row'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    if h!=v['sha256']:raise ValueError(f'corrupt receipt {path}')
    return v['row']


def certificate(root):
    h=hashed(root/'root.json');rows={};values={};masses={}
    def visit(k):
        if k in values:return values[k]
        r=hashed(root/'nodes'/f'{k}.json');rows[k]=r;n=r['model'];R=n['tau_R']
        if not n['R_safe'] or not r['R']['row']['success']:raise ValueError('R infeasible')
        if R!=r['R']['row']['elapsed_time']:raise ValueError('R time mismatch')
        actual_safe=len(r['cases'])==3 and all(c['task']['success'] and c['return_after_task'] is not None
                    and c['return_after_task']['row']['success'] for c in r['cases'])
        if actual_safe!=n['safe_C'] or actual_safe!=r['safe_C']:raise ValueError('admissibility mismatch')
        if actual_safe:
            by_child={c['child_id']:c for c in r['cases']}
            if len(n['children'])!=3:raise ValueError('missing child')
            for child in n['children']:
                if child['probability']!=1/3 or child['time']!=by_child[child['id']]['task']['elapsed_time']:
                    raise ValueError('transition mismatch')
        if not n['safe_C']:values[k]=(0.,R);return values[k]
        rewards={c['child_id']:c['task']['reward'] for c in r['cases']}
        outcomes=[(c,visit(c['id'])) for c in n['children']]
        A=sum(c['probability']*(rewards[c['id']]+v[0]) for c,v in outcomes)
        U=sum(c['probability']*(c['time']+v[1]) for c,v in outcomes)
        values[k]=(A,U);return A,U
    for r in h['roots']:visit(r['id'])
    first_rewards=[x['row']['reward'] for x in h['forced_first_tasks']]
    W=sum(r['probability']*(rew+values[r['id']][0]) for r,rew in zip(h['roots'],first_rewards))
    T=sum(r['probability']*(r['time']+values[r['id']][1]) for r in h['roots'])
    rho=W/T
    def occupy(k,p):
        masses[k]=masses.get(k,0.)+p
        for c in rows[k]['model']['children']:occupy(c['id'],p*c['probability'])
    for r in h['roots']:occupy(r['id'],r['probability'])
    safe=[]
    for k,(A,U) in values.items():
        n=rows[k]['model']
        if not n['safe_C']:continue
        R=n['tau_R'];delta=U-R
        dt=sum(c['probability']*(c['time']+rows[c['id']]['model']['tau_R']) for c in n['children'])-R
        if delta<=0:raise ValueError('positive tail duration required for rate-threshold corollary')
        safe.append(dict(node=k,sequence=rows[k]['descriptor']['sequence'],tail_reward=A,
            tail_extra_time=delta,tail_rate=A/delta,VB_excess=A-rho*delta,
            marginal_extra_time=dt,VB_visit_mass=masses[k]))
    if min(s['marginal_extra_time'] for s in safe)<0:
        raise ValueError('immediate-R duration is not certified to be a global lower bound')
    lower_T=sum(r['probability']*(r['time']+rows[r['id']]['model']['tau_R']) for r in h['roots'])
    D=sum(s['VB_visit_mass']*max(0.,-s['VB_excess']) for s in safe)
    upper=rho+D/lower_T
    tail_min=min(s['tail_rate'] for s in safe)
    beta=10.  # unchanged Config.recharge_overhead; used ONLY in analytic corollary
    inputs=[root/'root.json']+[root/'nodes'/f'{k}.json' for k in sorted(rows)]
    return dict(source='existing root and 36 cycle-node receipts only; no oracle result used',
        node_count=len(rows),safe_C_count=len(safe),VB_reward_per_cycle=W,VB_time_per_cycle=T,
        rho_VB=rho,T_min=lower_T,negative_tail_defect_load=D,rho_star_upper_bound=upper,
        VB_over_optimum_lower_bound=rho/upper,minimum_tail_rate=tail_min,
        additive_overhead_current=beta,additive_overhead_exact_VB_threshold=max(0.,W/tail_min-(T-beta)),
        overhead_threshold_scope='analytic fixed-transition-tree corollary only; no changed UAV run',
        negative_tail_states=[s for s in safe if s['VB_excess']<0],safe_states=safe,
        receipt_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=certificate(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['safe_states','receipt_sha256']},indent=2))

if __name__=='__main__':main()
