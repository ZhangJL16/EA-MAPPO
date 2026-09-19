"""T5.1d: exhaustive bounded measurement-history closure of existing V2 pilot.

The source archive is immutable. No generator, retention or hidden environment
is used. Depth is completed batches, not observations. All likelihoods exact.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic, process_time
import zipfile
import fcntl

from fpl.belief import PlannerState, outcomes
from fpl.evaluation import digest, save_new, source_hashes
from fpl.protocols import enumerate_protocols, PlanningLimit
from fpl.teacher_data import advance, exact_label, state_key, audit_label
from fpl.work import PlanningWorkBudget, WatchdogExpired
from fpl_v2.generator import from_json

ROOT = Path(__file__).resolve().parents[1]


def measuring_only(p, route):
    if not route.channels:
        return False
    ops = {o.name:o for o in p.operations}
    for name in route.operations:
        o = ops[name]
        if o.utility is None:
            if o.channels:return False  # Legacy observation-as-reward adapter.
        elif any(o.utility.by_hypothesis) or any(o.utility.observation_weights):
            return False
    return True


def closure(p, config):
    limits = config['closure_limits']
    work = PlanningWorkBudget(**{k:limits[k] for k in ('max_expansions','max_model_calls','watchdog_seconds')})
    start,cpu=monotonic(),process_time()
    root=PlannerState(p.prior,p.budget,p.capacity)
    rid=state_key(root)
    records={rid:dict(id=rid,state=dict(posterior=list(map(str,p.prior)),remaining=p.budget,resource=p.capacity),
                      depths=[0],provenance=[dict(source='measurement_closure',kind='root',history=[])])}
    front={rid:root};routes={};edges=[];expanded=[]
    status,reason='complete',None
    try:
        for depth in range(config['depth']):
            following={}
            for sid,state in sorted(front.items()):
                work.consume()
                if state.remaining not in routes:
                    routes[state.remaining]=[r for r in enumerate_protocols(p,state.remaining,limits['max_protocol_nodes'],work)
                                             if measuring_only(p,r)]
                for route in routes[state.remaining]:
                    branches=list(outcomes(p,state.posterior,route.channels,limits['max_outcomes'],work))
                    for feedback,mass,_ in branches:
                        child=advance(p,state,route,feedback);cid=state_key(child)
                        if cid not in records and len(records)>=limits['max_states']:
                            raise PlanningLimit('closure unique-state cap')
                        if len(edges)>=limits['max_edges']:
                            raise PlanningLimit('closure edge cap')
                        history=records[sid]['provenance'][0]['history']+[
                            dict(operations=list(route.operations),feedback=list(feedback))]
                        if cid not in records:
                            records[cid]=dict(id=cid,state=dict(posterior=list(map(str,child.posterior)),
                                remaining=child.remaining,resource=child.resource),depths=[],
                                provenance=[dict(source='measurement_closure',kind='reachable_feedback',history=history)])
                        if depth+1 not in records[cid]['depths']:records[cid]['depths'].append(depth+1)
                        following[cid]=child
                        edges.append(dict(depth=depth,source=sid,target=cid,operations=list(route.operations),
                                          feedback=list(feedback),mass=str(mass)))
                expanded.append([depth,sid])
            front=following
    except (PlanningLimit,WatchdogExpired) as exc:
        status,reason='incomplete',str(exc)
    return dict(status=status,reason=reason,records=[records[k] for k in sorted(records)],edges=edges,
                expanded=expanded,work=work.snapshot(),cpu_seconds=process_time()-cpu,wall_seconds=monotonic()-start)


def load_pilot(archive,plan):
    raw=Path(archive).read_bytes()
    if sha256(raw).hexdigest()!=plan['source_archive_sha256']:raise ValueError('wrong pilot archive')
    shards=[]
    with zipfile.ZipFile(archive) as z:
        hashes=json.loads(z.read('FILES_SHA256.json'))
        manifest=json.loads(z.read('admission/admission.json'))
        scheduled={Path(j['problem_file']).stem:j for j in manifest['jobs'] if j['pilot']}
        for name in z.namelist():
            if name.startswith('data/') and name.endswith('.shard.json'):
                content=z.read(name)
                if sha256(content).hexdigest()!=hashes[name]:raise ValueError('archive content mismatch')
                packed=json.loads(content);s=packed['payload']
                if digest(s)!=packed['hash']:raise ValueError('shard hash mismatch')
                stem=Path(name).name.removesuffix('.shard.json')
                if s['metadata']!=scheduled[stem]:raise ValueError('not a pilot variant')
                shards.append(s)
    if len(shards)!=32 or len({s['metadata']['clone_hash'] for s in shards})!=8:
        raise ValueError('wrong census scope')
    return sorted(shards,key=lambda s:from_json(s['problem']).instance_id)


def run(archive,plan,output,max_new=None,resume=False):
    shards=load_pilot(archive,plan);out=Path(output)
    seal=dict(plan=plan,source_hashes=source_hashes(),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              v2_generator_sha256=sha256((ROOT/'src/fpl_v2/generator.py').read_bytes()).hexdigest(),
              scheduled=[s['metadata'] for s in shards])
    if out.exists():
        if not resume or json.loads((out/'seal.json').read_text())!=seal:raise ValueError('resume seal mismatch')
    else:
        out.mkdir(parents=True);save_new(out/'seal.json',seal)
    with (out/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        count=0
        for old in shards:
            p=from_json(old['problem']);target=out/(p.instance_id+'.census.json')
            if target.exists():
                packed=json.loads(target.read_text())
                if packed['hash']!=digest(packed['payload']) or packed['payload']['seal_hash']!=digest(seal):
                    raise ValueError('completed census receipt mismatch')
                continue
            if max_new is not None and count>=max_new:break
            journal=out/(p.instance_id+'.started.json')
            if journal.exists():raise RuntimeError('interrupted variant; no automatic retry: '+p.instance_id)
            save_new(journal,dict(metadata=old['metadata'],status='started',retry=False))
            discovered=closure(p,plan)
            # Persist the entire label-independent closure BEFORE labels.
            save_new(out/(p.instance_id+'.closure.json'),discovered)
            records=[]
            oldrecords={r['id']:r for r in old['records']}
            for r in discovered['records']:
                s=r['state'];state=PlannerState(tuple(map(F,s['posterior'])),s['remaining'],s['resource'])
                label=exact_label(p,state,plan['label_limits'])
                if r['id'] in oldrecords and label['status']=='exact':
                    previous=oldrecords[r['id']]['label']
                    if previous['status']=='exact' and (label['value']!=previous['value'] or label['optimal_set']!=previous['optimal_set']):
                        raise AssertionError('existing exact label disagreement')
                records.append(dict(r,label=label,in_old_collector=r['id'] in oldrecords))
            payload=dict(metadata=old['metadata'],problem=old['problem'],seal_hash=digest(seal),
                         closure_hash=digest(discovered),closure_status=discovered['status'],records=records,
                         old_state_ids=sorted(oldrecords))
            save_new(target,dict(hash=digest(payload),payload=payload))
            count+=1
            print(json.dumps(dict(variant=p.instance_id,closure=discovered['status'],states=len(records),
                                  exact=sum(r['label']['status']=='exact' for r in records))),flush=True)


def independent_closure_check(p,plan,discovered):
    """Reconstruct reachability by normalized-belief breadth-first expansion.

    Not an independent model implementation: shares outcomes/route validation.
    Does not call ExactBayes or reuse the discovery's expansion decisions.
    """
    if discovered['status']!='complete':return dict(status='incomplete_not_certified')
    root=(p.budget,p.prior)
    front={root};nodes={root:{0}};expected_edges=set();expanded=set()
    for depth in range(plan['depth']):
        nxt=set()
        for H,b in front:
            sid=state_key(PlannerState(b,H,p.capacity));expanded.add((depth,sid))
            for route in enumerate_protocols(p,H,plan['closure_limits']['max_protocol_nodes']):
                if not measuring_only(p,route):continue
                for feedback,mass,post in outcomes(p,b,route.channels,plan['closure_limits']['max_outcomes']):
                    key=(H-route.duration,post);nxt.add(key)
                    nodes.setdefault(key,set()).add(depth+1)
                    cid=state_key(PlannerState(post,key[0],p.capacity))
                    expected_edges.add(digest(dict(depth=depth,source=sid,target=cid,operations=list(route.operations),
                                                  feedback=list(feedback),mass=str(mass))))
        front=nxt
    actual={r['id']:set(r['depths']) for r in discovered['records']}
    expected={state_key(PlannerState(b,H,p.capacity)):d for (H,b),d in nodes.items()}
    if actual!=expected or len(actual)!=len(discovered['records']):raise AssertionError('closure states/depths mismatch')
    actual_edges=[digest(e) for e in discovered['edges']]
    if set(actual_edges)!=expected_edges or len(actual_edges)!=len(expected_edges):raise AssertionError('closure edges mismatch')
    if set(map(tuple,discovered['expanded']))!=expanded:raise AssertionError('closure expansions mismatch')
    return dict(status='complete_checked',states=len(nodes),edges=len(expected_edges))


def characterize(p,r):
    label=r['label'];exact=label['status']=='exact'
    nonprior=tuple(map(F,r['state']['posterior']))!=p.prior
    required=False;optimal=False
    if exact:
        by={o.name:o.channels for o in p.operations}
        flags=[bool(path) and any(by[n] for n in path) for path in label['optimal_set']]
        required,optimal=all(flags),any(flags)
    return dict(states=1,exact=int(exact),unresolved=int(not exact),nonprior=int(nonprior),
                required=int(required),adaptive=int(nonprior and required),
                nonprior_measurement_optimal=int(nonprior and optimal),
                seen_old=int(r['in_old_collector']),new=int(not r['in_old_collector']),
                missed_adaptive=int(nonprior and required and not r['in_old_collector']))


def analyze(data,output,archive):
    data,out=Path(data),Path(output)
    if out.exists():raise FileExistsError(out)
    seal=json.loads((data/'seal.json').read_text());plan=seal['plan']
    old=load_pilot(archive,plan)
    expected={from_json(s['problem']).instance_id:s for s in old}
    if seal['source_hashes']!=source_hashes() or seal['script_sha256']!=sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('runtime seal mismatch')
    counters=defaultdict(Counter);groups=defaultdict(set);agroups=defaultdict(set);variants=defaultdict(set)
    depth_counts=defaultdict(Counter);witnesses=[];checks=[];work=Counter();hashes={};state_rows=[]
    files=sorted(data.glob('*.census.json'))
    if len(files)!=32:raise ValueError('census incomplete')
    for file in files:
        packed=json.loads(file.read_text());s=packed['payload'];p=from_json(s['problem']);meta=s['metadata']
        if digest(s)!=packed['hash'] or s['seal_hash']!=digest(seal):raise ValueError('receipt mismatch')
        if meta!=expected[p.instance_id]['metadata'] or s['problem']!=expected[p.instance_id]['problem']:
            raise ValueError('problem or scope changed')
        discovered=json.loads((data/(p.instance_id+'.closure.json')).read_text())
        if digest(discovered)!=s['closure_hash']:raise ValueError('discovery changed')
        if discovered['records']!=[{k:v for k,v in r.items() if k not in ('label','in_old_collector')} for r in s['records']]:
            raise ValueError('postlabel state selection')
        check=independent_closure_check(p,plan,discovered);checks.append(dict(variant=p.instance_id,**check))
        work['closure_cpu_seconds']+=discovered['cpu_seconds']
        work['closure_expansions']+=discovered['work']['expansions'];work['closure_model_calls']+=discovered['work']['model_calls']
        keys=[('ALL','ALL'),(meta['family'],'ALL'),('ALL',meta['fold']),(meta['family'],meta['fold'])]
        for k in keys:groups[k].add(meta['clone_hash']);variants[k].add(p.instance_id)
        oldids={r['id'] for r in expected[p.instance_id]['records']}
        for r in s['records']:
            if r['in_old_collector']!=(r['id'] in oldids):raise ValueError('collector membership mismatch')
            audit_label(p,r)
            cat=characterize(p,r)
            for k in keys:
                counters[k].update(cat)
                if cat['adaptive']:agroups[k].add(meta['clone_hash'])
            for depth in r['depths']:depth_counts[depth].update(cat)
            for metric in ('cpu_seconds','solver_states','likelihood_branches'):
                work['label_'+metric]+=r['label'][metric]
            for metric in ('expansions','model_calls'):work['label_'+metric]+=r['label']['work'][metric]
            row=dict(instance=p.instance_id,root=meta['root_seed'],family=meta['family'],fold=meta['fold'],
                     state=r['id'],depths=r['depths'],remaining=r['state']['remaining'],**cat)
            state_rows.append(row)
            if cat['adaptive']:
                witnesses.append(dict(row,posterior=r['state']['posterior'],value=r['label']['value'],
                                      optimal_set=r['label']['optimal_set'],history=r['provenance'][0]['history']))
        hashes[file.name]=sha256(file.read_bytes()).hexdigest()
    table=[dict(family=k[0],fold=k[1],groups=len(groups[k]),variants=len(variants[k]),
                adaptive_groups=len(agroups[k]),**counters[k]) for k in sorted(counters)]
    summary=dict(schema='teacher-v2-census-summary-v1',by_family_fold=table,
                 by_depth=[dict(depth=k,**v) for k,v in sorted(depth_counts.items())],
                 closure_checks=checks,work=dict(work),census_hashes=hashes,seal_hash=digest(seal),
                 scope='depth<=2 measurement-only complete batches; no claim about all reachable histories',
                 unchanged_objective=True,no_new_roots=True,no_training=True)
    out.mkdir(parents=True)
    for name,payload in [('summary.json',summary),('adaptive_witnesses.json',witnesses),('states.json',state_rows)]:save_new(out/name,payload)
    print(json.dumps(dict(by_family_fold=table,by_depth=summary['by_depth'],work=work),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','analyze'))
    parser.add_argument('--archive',required=True);parser.add_argument('--plan');parser.add_argument('--data')
    parser.add_argument('--output',required=True);parser.add_argument('--max-new',type=int);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.command=='run':run(args.archive,json.loads(Path(args.plan).read_text()),args.output,args.max_new,args.resume)
    else:analyze(args.data,args.output,args.archive)
