"""Read a completed P2-A2 result; never launch or wait for an experiment."""
import argparse
import json
from pathlib import Path
from research.regenerative_control.qualify import atomic


def summarize(r):
    if not r.get('complete_tree'):raise ValueError('complete exact tree required')
    early=[];loss=0.
    for key,meta in r['node_metadata'].items():
        if meta['origin']!='cycle':continue  # nominated-state duplicate is not population evidence
        delta=r['delta_Q'][key]
        vb_mass=r['VB_occupancy'].get(key,0.)
        if delta is not None and delta<0:
            loss-=vb_mass*delta
        if delta is not None and delta < -1e-9:
            early.append(dict(node=key,sequence=meta['sequence'],state=meta['state'],delta_Q=delta,
                optimal_visit_probability=r['optimal_occupancy'].get(key,0.),
                VB_visit_probability=vb_mass,VB_excess_loss=-vb_mass*delta))
    expected=r['rho_star']*r['VB_time_per_cycle']-r['VB_tasks_per_cycle']
    if abs(loss-expected)>1e-8*max(1,abs(expected)):
        raise ValueError('independent policy-excess loss decomposition failed')
    for e in early:e['fraction_of_excess_loss']=e['VB_excess_loss']/loss if loss>1e-12 else None
    physical={json.dumps(e['state'],sort_keys=True) for e in early}
    return dict(decision=r['decision'],rho_star=r['rho_star'],rho_VB=r['rho_VB'],
        VB_over_oracle=r['VB_over_oracle'],ratio_interval=r['ratio_interval'],counts=r['counts'],
        safe_return_optimal_cycle_prefixes=len(early),distinct_physical_early_return_states=len(physical),
        early_return_nodes_reached_by_optimal=sum(e['optimal_visit_probability']>0 for e in early),
        optimal_probability_of_early_return_per_cycle=sum(e['optimal_visit_probability'] for e in early),
        VB_expected_visits_to_early_return_states=sum(e['VB_visit_probability'] for e in early),
        VB_excess_loss=loss,independent_loss_identity_residual=loss-expected,
        largest_single_prefix_loss_fraction=max((e['fraction_of_excess_loss'] or 0 for e in early),default=0),
        early_return_states=early,candidate=r['candidate'],scope=r['scope'],
        promotion='No automatic promotion. KILL stops main story; a gap still needs mechanism and non-pathology review.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    if not (args.input/'result.json').exists():raise SystemExit('No completed result; no ratio/kill conclusion available.')
    result=json.loads((args.input/'result.json').read_text());summary=summarize(result)
    atomic(args.input/'structural_summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
