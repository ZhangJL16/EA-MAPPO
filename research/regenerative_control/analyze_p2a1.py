"""Descriptive branch census, closure, abstraction and local cycle contrast.

Figure contract: three quantitative scatter panels show whether battery/mission
margins explain the *fixed bounded-cycle contrast*, not Q*. All collected states
remain in source data; undefined contrast states are explicitly counted. Python
matplotlib, editable SVG/PDF plus PNG previews, 90x75 mm individual panels.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from research.regenerative_control.census_p2a1 import digest
from research.regenerative_control.qualify import atomic


def threshold_fit(x,labels):
    x=np.array(x);y=np.array(labels,dtype=bool)
    if not len(x): return None
    candidates=np.r_[-np.inf,np.unique(x),np.inf]
    acc=[np.mean((x>t)==y) for t in candidates]
    k=int(np.argmax(acc));t=float(candidates[k])
    return dict(n=len(x),accuracy=float(acc[k]),threshold=t if np.isfinite(t) else str(t),
                positive=int(y.sum()),negative=int((~y).sum()),
                constant_accuracy=float(max(y.mean(),1-y.mean())),fixed_zero_accuracy=float(np.mean((x>0)==y)),resubstitution_only=True)


def abstractions(rows):
    reports=[]
    for name in ['z0_e','z1_e_distance','z2_e_distance_speed','z3_e_position_velocity']:
        bins=defaultdict(list)
        for r in rows:
            s=r['state']['state'];p=np.array(s['position']);v=np.array(s['velocity'])
            e=int(np.floor(s['energy']/2));d=int(np.floor(np.linalg.norm(p-[2880,2000,200])/10))
            if name=='z0_e': key=(e,)
            elif name=='z1_e_distance':key=(e,d)
            elif name=='z2_e_distance_speed': key=(e,d,int(np.floor(np.linalg.norm(v))))
            else:key=(e,*np.floor(p/10).astype(int),*np.floor(v).astype(int))
            bins[key].append(r)
        detail=[]
        for key,g in bins.items():
            if len(g)<3:continue
            er=[];tr=[];ev=[];tv=[];mix=[];transitions={}
            for branch in ['C100','C300','C600','R']:
                b=[r['branches'][branch]['row'] for r in g]
                energy=[x['energy_used'] for x in b];time=[x['elapsed_time'] for x in b]
                er.append(float(np.ptp(energy)));tr.append(float(np.ptp(time)))
                ev.append(float(np.var(energy)));tv.append(float(np.var(time)))
                mix.append(len({x['success'] for x in b})>1)
                nxt=[x['next_macro_state'] for x in b]
                transitions[branch]=dict(
                    energy_range=float(np.ptp([x['energy'] for x in nxt])),
                    position_coordinate_ranges=np.ptp([x['position'] for x in nxt],axis=0).tolist(),
                    velocity_coordinate_ranges=np.ptp([x['velocity'] for x in nxt],axis=0).tolist(),
                    phase_counts={p:sum(x['phase']==p for x in nxt) for p in {x['phase'] for x in nxt}})
            detail.append(dict(bin=list(map(int,key)),n=len(g),max_energy_range=max(er),
                max_time_range=max(tr),max_energy_variance=max(ev),max_time_variance=max(tv),
                successor_spread=transitions,mixed_success=any(mix),stable=max(er)<=1 and max(tr)<=5 and not any(mix)))
        covered=sum(g['n'] for g in detail);stable=sum(g['n'] for g in detail if g['stable'])
        reports.append(dict(abstraction=name,total_bins=len(bins),tested_bins=len(detail),
            tested_states=covered,coverage=covered/len(rows),stable_states=stable,
            stable_fraction_among_tested=stable/covered if covered else None,
            mixed_success_bins=sum(g['mixed_success'] for g in detail),
            adequate_support=covered/len(rows)>=.5 and len(detail)>=20,
            stable_at_declared_tolerances=bool(covered/len(rows)>=.5 and len(detail)>=20 and stable==covered),
            bins=detail))
    return reports


def pairs(data,margin):
    counts=dict(comparable_pairs=0,opposite_sign_pairs=0,velocity_matched_pairs=0,
                velocity_matched_opposite=0)
    examples=[]
    for a,b in itertools.combinations(data,2):
        if a['trajectory_id']==b['trajectory_id']:continue
        if abs(a['energy']-b['energy'])>=2 or abs(a[margin]-b[margin])>=1 or abs(a['home_distance']-b['home_distance'])>=10:continue
        if abs(a['local_delta'])<=1e-6 or abs(b['local_delta'])<=1e-6:continue
        counts['comparable_pairs']+=1
        flip=a['local_delta']*b['local_delta']<0
        counts['opposite_sign_pairs']+=int(flip)
        if np.linalg.norm(np.array(a['velocity'])-b['velocity'])<.5:
            counts['velocity_matched_pairs']+=1;counts['velocity_matched_opposite']+=int(flip)
        if flip: examples.append([a['state_id'],b['state_id']])
    counts['fraction']=(counts['opposite_sign_pairs']/counts['comparable_pairs'] if counts['comparable_pairs'] else None)
    return dict(**counts,opposite_pairs=examples,pairs_not_independent=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    src,out=args.input,args.output;out.mkdir(parents=True,exist_ok=True)
    census=json.loads((src/'census.json').read_text());states=census['states']
    ref=json.loads((src/'after1_reference.json').read_text());rho=ref['rho_after1']
    rows=[]
    for meta in states:
        envelope=json.loads((src/'branches'/f"{meta['state_id']}.json").read_text())
        assert digest(envelope['row'])==envelope['sha256']
        rows.append(envelope['row'])
    support={m['state_id'] for m in states}
    closure=defaultdict(int);table=[];data=[]
    for branch in ['C100','C300','C600','R']:
        rr=[r['branches'][branch]['row'] for r in rows]
        table.append(dict(branch=branch,attempted=len(rr),success=sum(x['success'] for x in rr),
            failure=sum(not x['success'] and not x['censored'] for x in rr),
            censored=sum(x['censored'] for x in rr),
            collision=sum(x['collision'] for x in rr),
            energy_depletion=sum(x['terminal_reason']=='energy_depletion' for x in rr),
            navigation_timeout=sum(x['timeout'] for x in rr)))
    for r in rows:
        for branch,b in r['branches'].items():
            if not b['row']['success']:closure['failure_absorbing']+=1
            elif branch=='R':closure['canonical_H']+=1
            else:
                closure['successful_D_successors']+=1
                closure['represented_D_successors']+=int(digest(b['end_physical']) in support)
        m=r['state'];s=m['state'];e=s['energy'];R=r['branches']['R']['row']
        costs=[r['missions'][str(d)]['requirement']['energy'] for d in (100,300,600)]
        viable=R['success'] and all(r['missions'][str(d)]['actual']['success'] for d in (100,300,600))
        local=None
        if viable:
            t=np.mean([r['missions'][str(d)]['requirement']['time'] for d in (100,300,600)])
            local=float(1-rho*(t-R['elapsed_time']))
        finite=all(c is not None for c in costs)
        data.append(dict(state_id=m['state_id'],trajectory_id=m['trajectory_id'],energy=e,
            position=s['position'],velocity=s['velocity'],previous_task_type=m['previous_task_type'],
            source_policy=m['occupancy_source_policy'],previous_contact=s['previous_contact'],
            home_distance=float(np.linalg.norm(np.array(s['position'])-[2880,2000,200])),
            mission_energy_mean=float(np.mean(costs)) if finite else None,
            mission_energy_max=max(costs) if finite else None,
            m_mean=float(e-np.mean(costs)) if finite else None,m_max=float(e-max(costs)) if finite else None,
            local_delta=local,all_cycle_alternatives_viable=bool(viable),R_viable=R['success'],
            local_action=(('C' if local>0 else 'R') if viable else ('R' if R['success'] else None)),
            energy_interventions=sum(r['missions'][str(d)]['requirement']['counterfactual_energy_intervention'] for d in (100,300,600))))
    valid=[r for r in data if r['local_delta'] is not None and r['m_mean'] is not None]
    labeled=[r for r in data if r['local_action'] is not None and r['m_mean'] is not None]
    fits={}
    for field in ['energy','m_mean','m_max']:
        fits[field]=dict(common_viable=threshold_fit([r[field] for r in valid],[r['local_delta']>0 for r in valid]),
            viability_constrained=threshold_fit([r[field] for r in labeled],[r['local_action']=='C' for r in labeled]))
    ab=abstractions(rows)
    closure['closure_rate_D']=closure['represented_D_successors']/max(1,closure['successful_D_successors'])
    # H outcomes use a separate explicit enumeration; exact physical key below.
    h_closed=[]
    for b in ref['branches']:
        q=b['task']['next_macro_state']
        h_closed.append(any(m['state']['position']==q['position'] and m['state']['velocity']==q['velocity'] and m['state']['energy']==q['energy'] for m in states))
    closure['H_forced_pose_velocity_energy_support']=h_closed
    summary=dict(n_states=len(states),n_occupancy_occurrences=census['occurrences'],n_source_trajectories=len(census['trajectories']),
        energy_range=[min(r['energy'] for r in data),max(r['energy'] for r in data)],
        position_min=np.min([r['position'] for r in data],axis=0).tolist(),position_max=np.max([r['position'] for r in data],axis=0).tolist(),
        speed_range=[float(min(np.linalg.norm(r['velocity']) for r in data)),float(max(np.linalg.norm(r['velocity']) for r in data))],
        velocity_min=np.min([r['velocity'] for r in data],axis=0).tolist(),velocity_max=np.max([r['velocity'] for r in data],axis=0).tolist(),
        policies={p:sum(r['source_policy']==p for r in data) for p in sorted({r['source_policy'] for r in data})},
        source_failure_trajectories=sum(t['audit']['failed'] for t in census['trajectories']),
        branch_table=table,closure=dict(closure),rho_after1=rho,common_viable_states=len(valid),
        undefined_local_delta=len(data)-len(valid),local_delta_range=[min(r['local_delta'] for r in valid),max(r['local_delta'] for r in valid)] if valid else None,
        mission_interventions=sum(r['energy_interventions'] for r in data),
        abstraction=[{k:v for k,v in a.items() if k!='bins'} for a in ab],
        fits=fits,sign_pairs={m:pairs(valid,m) for m in ['m_mean','m_max']},
        oracle_solved=False,rho_star=None,neural_updates=0)
    atomic(out/'summary.json',summary);atomic(out/'state_values.json',data);atomic(out/'abstraction_bins.json',ab)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    for field,label in [('energy','Battery energy'),('m_mean','Mean mission reserve'),('m_max','Max mission reserve')]:
        fig,ax=plt.subplots(figsize=(3.5,3.2),layout='constrained')
        for task,color,marker in [(0,'#4477AA','o'),(1,'#228833','s'),(2,'#AA3377','^')]:
            g=[r for r in valid if r['previous_task_type']==task]
            ax.scatter([r[field] for r in g],[r['local_delta'] for r in g],s=12,alpha=.7,color=color,marker=marker,
                       label=f'Previous task {(100,300,600)[task]}')
        ax.axhline(0,color='0.35',linewidth=.7)
        ax.set(xlabel=label,ylabel='Local cycle contrast (not Q*)',title=f'Common viable states: {len(valid)}/{len(data)}\nUndefined contrast: {len(data)-len(valid)}')
        ax.legend(fontsize=6,loc='best')
        fig.savefig(out/f'{field}_vs_local_delta.svg',bbox_inches='tight')
        svg=out/f'{field}_vs_local_delta.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        fig.savefig(out/f'{field}_vs_local_delta.pdf',bbox_inches='tight')
        fig.savefig(out/f'{field}_vs_local_delta.png',dpi=300,bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
