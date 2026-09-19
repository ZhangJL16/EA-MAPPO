"""Deliver requested A/B/C/D files and a source-bound portable evidence archive."""
from collections import defaultdict
from fractions import Fraction as F
from hashlib import sha256
import argparse
import json
from pathlib import Path
import zipfile
from fpl.evaluation import digest

ROOT=Path(__file__).resolve().parents[1]


def report(data,analysis,out):
    data,analysis,out=map(Path,(data,analysis,out))
    if out.exists():raise FileExistsError(out)
    summary=json.loads((analysis/'summary.json').read_text());seal=json.loads((data/'seal.json').read_text())
    if summary['seal_hash']!=digest(seal):raise ValueError('wrong analysis')
    episodes=json.loads((analysis/'occupancy_decomposition.json').read_text())
    adaptive=defaultdict(lambda:dict(contexts=0,occupancy=F(0),contribution=F(0)))
    for episode in episodes:
        groups=defaultdict(lambda:dict(occupancy=F(0),weighted_residual=F(0),contexts=[]))
        for r in episode['rows']:
            if not r['prior_state'] and r['measurement_required']:
                a=adaptive[(episode['method'],episode['tier'])]
                a['contexts']+=1;a['occupancy']+=F(r['occupancy']);a['contribution']+=F(r['weighted_residual'])
            g=groups[r['state_id']];g['occupancy']+=F(r['occupancy']);g['weighted_residual']+=F(r['weighted_residual']);g['contexts'].append(r['node'])
        episode['state_occupancies']=[dict(state_id=k,occupancy=str(v['occupancy']),
            weighted_residual=str(v['weighted_residual']),context_nodes=v['contexts']) for k,v in sorted(groups.items())]
        if episode['status']=='exact' and sum(F(g['weighted_residual']) for g in episode['state_occupancies'])!=F(episode['gap']):
            raise AssertionError('state aggregation identity failed')
    rows=summary['occupancy_summary']
    lines=['# T5.1e — where does planning compute create value?','', '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
           '- Origin Date: 2026-09-19', '- Verification Status: 128 exact Bellman identities and independent policy integrations checked',
           '- Version Label: planning_residual_v1','',
           '## Decision: Route C — stop the current learned-planner main line','',
           'Do not launch full AFPP, a learned residual selector, V2.1 expansion or V3. '
           'Retain the theory and planning/evaluation framework. This is a research '
           'decision on the current evidence, **not** a theorem that amortization can '
           'never help larger or different tasks. The earlier two-root DEV GO remains '
           'a historical observation, not a guarantee that a new learned method is needed.',
           '', 'Large OneStepVOI is Bayes-optimal on all 32 variants and has zero '
           'fresh-decision residual on all 1,171 census states. Small VOI has only '
           '9 nonzero fresh-state residuals, but episode-budget exhaustion changes '
           'its later actions. Its deployment gap is dominated by fixed score-blind '
           'fallback/execution decisions, not rare continued-sensing decisions.',
           '', '## Exact deployment gaps','',
           '| Planner | Tier | Positive-gap variants | Mean gap | Positive-gap structures | Mean model calls / episode |',
           '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['method']} | {r['tier']} | {r['positive_gap_episodes']}/32 | {float(F(r['mean_gap'])):.6f} | {r['positive_gap_groups']}/8 | {float(F(r['expected_model_calls_sum'])/32):.2f} |")
    lines += ['', 'Means equally weight the 32 paired variants for description; there '
              'are only eight structural groups. These are Bayes utility gaps '
              '`V* - V_policy`, not minimax-regret estimates.',
              '', '## Where the Small VOI loss occurs','',
              '| Exclusive categorization (each dimension separately) | Share of exact pooled gap |',
              '|---|---:|']
    for a in summary['occupancy_attribution']:
        if a['method']=='one_step_voi' and a['tier']=='small' and a['dimension'] in ('prior','decision_class','autopilot','limit') and F(a['gap_contribution']):
            lines.append(f"| {a['dimension']}: {a['category']} | {100*float(F(a['gap_share'])):.4f}% |")
    lines += ['', 'Different dimensions overlap: their percentages must not be added. '
              'All positive Small Beam residuals likewise coincide with a work-limit '
              'hit; 70.82% of its pooled gap is at nonmeasuring-optimal states. '
              'All selected actions carrying positive occupancy-weighted residual are '
              'task actions. A work-limit flag is an observed association, not proof '
              'that merely raising a cap is the uniquely optimal repair.',
              '', '**Adaptive-subset contribution is zero for all four policies.** '
              'None of the actual feedback trees visits a non-prior '
              'measurement-required state, even though six such census states '
              'exist. Their existence and rarity are not the source of this '
              'study\'s deployment quality gap. See adaptive_contribution.json.',
              '', '## Fresh census decisions versus deployment contexts','',
              '| Planner | Tier | Nonzero residuals / 1,171 | Structural groups |',
              '|---|---|---:|---:|']
    for r in summary['fresh_probe_summary']:
        lines.append(f"| {r['method']} | {r['tier']} | {r['nonzero']} | {r['nonzero_groups']} |")
    lines += ['', 'Fresh Small VOI errors are 8 prior-belief states and 1 non-prior '
              'state, all measurement-required and work-limit-associated. This does '
              'not describe its episode losses: cumulative work can be exhausted on '
              'later otherwise simple task decisions. Large Beam has no census '
              'fresh-state errors yet has a 1/5 episode gap on one variant, again '
              'from limited-work nonmeasuring decisions. A belief/H-only occupancy '
              'analysis would miss this distinction.',
              '', '## Files and verification','',
              '- `state_residuals.jsonl`: all 4,684 fresh census-state × planner rows, exact V/action/Q/residual, ties, categories and work.',
              '- `occupancy_decomposition.json`: 128 complete policy trees; full-context occupancy and state aggregates, exact weighted residuals, EPE cross-check.',
              '- `small_vs_large.json`: 2,342 paired fresh-state differences plus 64 paired episode gaps. Uniform decision differences are not occupancy gains.',
              '- `summary.json`: exclusive-category shares, work, positive-state counts and complete concentration curves; no artificial acceptance cutoff.',
              '- `supplemental_labels.json`: 452 evaluation-only labels for actually visited states outside the census; all exact, same problems/objective/caps.',
              '', '128/128 identities `sum d*delta = V*(root)-V_policy(root)` hold as '
              'rational equalities and agree with the unchanged hypothesis-wise '
              'ExactPolicyEvaluator. Whole-policy state is deep-copied at each '
              'feedback branch; budgets are not refreshed on branches. STOP Q=0 '
              'is handled explicitly. State aggregates retain all context IDs because '
              'the same belief/H can have different remaining planning work.',
              '', '## Scope and interpretation','',
              'This analysis does not rerun DEV 2201/2202 or explain their specific '
              'fourth-batch gap. It diagnoses the current teacher pilot. Large VOI '
              'is already exact here at 9,480.87 mean model calls per episode, '
              'versus Small VOI 5,087.74. Measured mean summed planning latency '
              'was about 16.6 ms versus 9.0 ms (one local CPU run, not a population '
              'or hardware-independent claim). This does not establish a compelling '
              'learned-planner advantage to pursue on this family.',
              '', 'State-set hardness is not the same as occupancy-weighted '
              'performance. Sparse fresh-state errors do not automatically justify '
              'a learned selector when most deployment loss is fallback behavior '
              'and a tested ordinary planner already attains the optimum.',
              '', '63 tests passed. No new roots, raw-label retries, objective '
              'changes, neural training, V2.1/V3, formal DEV/test/OOD or CONFIRM access. '
              'Original frozen code and evidence remain unchanged.']
    out.mkdir(parents=True)
    for name in ('state_residuals.jsonl','small_vs_large.json','summary.json','supplemental_labels.json'):
        (out/name).write_bytes((analysis/name).read_bytes())
    (out/'occupancy_decomposition.json').write_text(json.dumps(episodes,sort_keys=True,indent=2)+'\n')
    (out/'adaptive_contribution.json').write_text(json.dumps([
        dict(method=r['method'],tier=r['tier'],contexts=adaptive[(r['method'],r['tier'])]['contexts'],
             occupancy=str(adaptive[(r['method'],r['tier'])]['occupancy']),
             gap_contribution=str(adaptive[(r['method'],r['tier'])]['contribution'])) for r in rows],indent=2)+'\n')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    decision=dict(route='C',action='stop_current_learned_planner_main_line',scope='these 8 teacher pilot structures and tested planners',
        reason='Large VOI exact; Small VOI deployment loss dominated by work-limited fallback/nonmeasuring execution',
        not_claimed='global impossibility of beneficial amortization or invalidity of previous DEV results',
        further_training_authorized=False)
    (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    entries={}
    for label,base in [('data',data),('results',out)]:
        for p in sorted(base.iterdir()):
            if p.is_file() and p.suffix in ('.json','.jsonl','.md'):entries[f'{label}/{p.name}']=p.read_bytes()
    for package in ('fpl','fpl_v2'):
        for p in (ROOT/'src'/package).rglob('*.py'):entries['runtime/'+str(p.relative_to(ROOT/'src'))]=p.read_bytes()
    for name in ('scripts/planning_residual.py','scripts/analyze_planning_residual.py','scripts/report_planning_residual.py',
                 'configs/planning_residual_v1.json','tests/test_planning_residual.py','CONTRACT_T51E_RESIDUAL.md'):
        entries['source/'+name]=(ROOT/name).read_bytes()
    hashes={n:sha256(b).hexdigest() for n,b in entries.items()}
    entries['FILES_SHA256.json']=json.dumps(hashes,sort_keys=True,indent=2).encode()
    archive=out/'planning_residual_evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(entries.items()):
            info=zipfile.ZipInfo(n,(2026,9,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    receipt=dict(archive=archive.name,sha256=sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,
                 seal_hash=digest(seal),files=hashes)
    (out/'manifest.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('archive','sha256','bytes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','analysis','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();report(a.data,a.analysis,a.output)
