"""Pure TRAIN target derivation from an audited archive. No FPL/solver imports."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import zipfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def action_key(kind, name=None):
    return json.dumps([kind, name], separators=(',', ':'))


STOP = action_key('stop')
END = action_key('end')


def prefix_position(problem, names, H):
    """Decode-only position; pending observations never update the belief."""
    node, resource, elapsed = problem['reset'], problem['capacity'], 0
    counts = Counter()
    ops = {op['name']:op for op in problem['operations']}
    for i,name in enumerate(names):
        op = ops[name]
        if op['source'] != node or op['energy'] > resource:
            raise ValueError('illegal prefix')
        resource -= op['energy']
        elapsed += op['duration']
        counts.update(op['channels'])
        node = op['target']
        if (elapsed > H or sum(counts.values()) > problem['max_measurements'] or
            any(n > problem['per_channel_limit'] for n in counts.values())):
            raise ValueError('prefix violates budget/measurement limits')
        if node == problem['reset'] and i != len(names)-1:
            raise ValueError('protocol passes its first reset return')
    complete = bool(names) and node == problem['reset']
    return dict(node=node, elapsed=elapsed, remaining_deployment_budget=H-elapsed,
                remaining_resource=resource, pending_channel_counts=dict(sorted(counts.items())),
                complete=complete, resource_after_return=problem['capacity'] if complete else None)


def derive_prefixes(problem, record):
    label = record['label']
    if label['status'] != 'exact':
        return []
    H = record['state']['remaining']
    branches = defaultdict(dict)
    stop_seen, seen = False, set()
    for candidate in label['candidates']:
        names, q = candidate['operations'], F(candidate['q'])
        if names is None:
            if stop_seen or q != 0:
                raise ValueError('invalid stop target')
            stop_seen = True
            branches[()][STOP] = q
            continue
        names = tuple(names)
        if not names or names in seen:
            raise ValueError('duplicate/empty complete protocol')
        seen.add(names)
        if not prefix_position(problem,names,H)['complete']:
            raise ValueError('protocol does not return to reset')
        for depth,name in enumerate(names):
            d = branches[names[:depth]]
            key = action_key('operation', name)
            d[key] = max(d[key],q) if key in d else q
        branches[names][END] = q
    if not stop_seen or max(branches[()].values()) != F(label['value']):
        raise ValueError('root V/Q/stop mismatch')
    decisions = sum(not prefix_position(problem,u,H)['complete'] for u in branches)
    rows = []
    for u, options in sorted(branches.items()):
        position = prefix_position(problem,u,H)
        if position['complete'] and set(options) != {END}:
            raise ValueError('complete protocol has a continuation operation')
        if u and not position['complete'] and (STOP in options or END in options):
            raise ValueError('mid-protocol stop is not legal')
        v = max(options.values())
        rows.append(dict(prefix=list(u), position=position,
                         legal_next=sorted(options), q_next={k:str(vv) for k,vv in sorted(options.items())},
                         advantage={k:str(vv-v) for k,vv in sorted(options.items())},
                         optimal_next=sorted(k for k,vv in options.items() if vv==v),
                         value_prefix=str(v), normalized_value_prefix=str(v/max(H,1)),
                         policy_weight=str(F(0) if position['complete'] else F(1,decisions)),
                         loss_mask=not position['complete']))
    return rows


def verify_prefixes(problem, record, rows):
    """Reference scan over saved candidates, independent of trie aggregation."""
    candidates = record['label']['candidates']
    expected = {()}
    for c in candidates:
        if c['operations']:
            p = tuple(c['operations'])
            expected.update(p[:j] for j in range(len(p)+1))
    if len(rows)!=len(expected) or {tuple(r['prefix']) for r in rows} != expected:
        raise ValueError('prefix coverage mismatch')
    for r in rows:
        u = tuple(r['prefix'])
        scores = defaultdict(list)
        for c in candidates:
            p = None if c['operations'] is None else tuple(c['operations'])
            if p is None:
                if not u:
                    scores[STOP].append(F(c['q']))
            elif p[:len(u)] == u:
                scores[END if p==u else action_key('operation',p[len(u)])].append(F(c['q']))
        targets = {k:max(v) for k,v in scores.items()}
        v = max(targets.values())
        if (set(r['legal_next']) != set(targets) or {k:F(x) for k,x in r['q_next'].items()} != targets
            or r['optimal_next'] != sorted(k for k,x in targets.items() if x==v)
            or {k:F(x) for k,x in r['advantage'].items()} != {k:x-v for k,x in targets.items()}
            or F(r['value_prefix']) != v
            or F(r['normalized_value_prefix']) != v/max(record['state']['remaining'],1)
            or r['position'] != prefix_position(problem,u,record['state']['remaining'])):
            raise ValueError('prefix target mismatch')
    if sum(F(r['policy_weight']) for r in rows) != 1:
        raise ValueError('unequal total state weight')
    if any(F(r['policy_weight']) != 0 for r in rows if r['position']['complete']):
        raise ValueError('terminal decoder rows must be loss-masked')


def group_folds(shards, salt):
    """Metadata-only split; never look at Q or adaptive-state availability."""
    roots = defaultdict(set)
    groups = set()
    for s in shards:
        group = s['metadata']['clone_hash']
        groups.add(group)
        roots[s['metadata']['root_seed']].add(group)
    if any(len(v)!=1 for v in roots.values()):
        raise ValueError('one root spans structural groups: connected-component grouping required')
    if len(groups)<2:
        return {g:'student_train' for g in groups}
    ordered = sorted(groups,key=lambda g:digest([salt,g]))
    count = max(1,len(groups)//4)
    return {g:('student_validation' if g in ordered[:count] else 'student_train') for g in groups}


def load_audited(archive, audit_path):
    audit = json.loads(Path(audit_path).read_text())
    raw = Path(archive).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=audit['archive_sha256'] or audit['status']!='passed':
        raise ValueError('input archive is not the audited artifact')
    if audit['summary']['status']!='complete':
        raise ValueError('complete input audit required')
    with zipfile.ZipFile(archive) as z:
        seal = json.loads(z.read('data/seal.json'))
        jobs = {j['instance_hash']:j['metadata'] for j in seal['jobs']}
        shards = []
        for name in sorted(n for n in z.namelist() if n.endswith('.shard.json') and n.startswith('data/')):
            envelope = json.loads(z.read(name))
            s = envelope['payload']
            if (digest(s)!=envelope['hash'] or s['instance_hash'] not in jobs
                or s['metadata']!=jobs.pop(s['instance_hash']) or s['problem']['split_id']!='train'
                or digest(s['problem'])!=s['instance_hash']):
                raise ValueError('shard identity/hash/split mismatch')
            if len({r['id'] for r in s['records']})!=len(s['records']):
                raise ValueError('duplicate state identity')
            shards.append(s)
        if jobs:
            raise ValueError('missing shards')
    return seal, shards


def write_json(path, data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')


def export(archive, audit_path, config_path, output):
    conf = json.loads(Path(config_path).read_text())
    if conf['schema']!='teacher-prefix-export-v1':
        raise ValueError('unsupported export contract')
    fixed = dict(normalization='divide by max(original decision H,1), including prefix value',
                 terminal_decoding='END only, policy loss masked; STOP at empty prefix only',
                 class_balancing='none in this export',
                 sampling='uniform state within its fold, then uniform nonterminal prefix within state; action loss averaged within prefix',
                 folds='metadata-only salted hash order; floor(groups/4) validation groups, at least one if groups >=2')
    if any(conf.get(k)!=v for k,v in fixed.items()):
        raise ValueError('unsupported normalization/sampling/terminal/fold semantics')
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest()!=conf['input_archive_sha256']:
        raise ValueError('wrong frozen corpus')
    out = Path(output)
    if out.exists():
        raise FileExistsError('new output required; export never overwrites')
    seal, shards = load_audited(archive,audit_path)
    folds = group_folds(shards,conf['group_split_salt'])
    problems, states, prefixes, excluded = {}, [], [], []
    summary = Counter()
    per_fold = defaultdict(Counter)
    for s in shards:
        pid, group = s['instance_hash'], s['metadata']['clone_hash']
        problems[pid] = {k:v for k,v in s['problem'].items() if k not in ('instance_id','family_id','split_id')}
        for r in s['records']:
            sid = digest([pid,r['id']])
            metadata = dict(root=s['metadata']['root_seed'],condition=s['metadata']['condition'],
                            structural_group=group,fold=folds[group],sources=sorted({x['source'] for x in r['provenance']}),
                            solver_status=r['label']['status'], solver_states=r['label']['solver_states'],
                            work=r['label']['work'], cpu_seconds=r['label']['cpu_seconds'])
            if r['label']['status']!='exact':
                excluded.append(dict(id=sid,metadata=metadata,reason=r['label'].get('reason')))
                continue
            pp = derive_prefixes(s['problem'],r)
            verify_prefixes(s['problem'],r,pp)
            normalized = str(F(r['label']['value'])/max(r['state']['remaining'],1))
            state = dict(id=sid,inputs=dict(problem_id=pid,posterior=r['state']['posterior'],
                                           remaining=r['state']['remaining'],capacity=r['state']['resource']),
                         targets=dict(value=r['label']['value'],normalized_value=normalized,
                                      optimal_protocols=r['label']['optimal_set'],protocols=r['label']['candidates']),
                         metadata=metadata,prefix_count=len(pp),value_weight='1')
            states.append(state)
            prefixes.extend(dict(id=digest([sid,p['prefix']]),state_id=sid,**p) for p in pp)
            opts = [c for c in r['label']['candidates'] if c['operations'] in r['label']['optimal_set']]
            nonprior = r['state']['posterior']!=s['problem']['prior']
            required = all(bool(c.get('channels')) for c in opts)
            per_fold[folds[group]].update(states=1,nonprior_required=int(nonprior and required))
            summary.update(states=1,prefixes=len(pp),decision_prefixes=sum(p['loss_mask'] for p in pp),
                           tied_decision_prefixes=sum(p['loss_mask'] and len(p['optimal_next'])>1 for p in pp))
    for fold in set(folds.values()):
        per_fold[fold]['structural_groups'] = sum(v==fold for v in folds.values())
    out.mkdir(parents=True)
    write_json(out/'problems.json', problems)
    write_json(out/'group_folds.json',dict(salt=conf['group_split_salt'],groups=folds,
                                       scope='internal TRAIN-only diagnostic holdout; not DEV/test/confirmation'))
    for name,rows in [('states',states),('prefixes',prefixes),('excluded',excluded)]:
        with (out/(name+'.jsonl')).open('w') as f:
            for row in rows:
                f.write(json.dumps(row,sort_keys=True)+'\n')
    files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    manifest = dict(schema='teacher-target-export-v1',status='complete',summary=dict(summary),
                    excluded_states=len(excluded),fold_summary={k:dict(v) for k,v in per_fold.items()},
                    input_archive_sha256=conf['input_archive_sha256'],input_audit_sha256=hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
                    export_config=conf,exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=files,
                    verification='all prefix Q/advantage/set targets checked by direct saved-candidate scan',
                    solver_calls=0,environment_calls=0,
                    limits='No new information or structures; inherited V1 coverage. Fold assignment not balanced by outcomes.')
    write_json(out/'manifest.json',manifest)  # Completion marker last; partial directories are not datasets.
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive',required=True)
    parser.add_argument('--audit',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.archive,args.audit,args.config,args.output),indent=2))
