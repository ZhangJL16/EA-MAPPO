import itertools
import numpy as np


def generate(config):
    jobs=[]
    def add(suite, method, seed, truth, horizon, **spec):
        jobs.append(dict(job_id=f'j{len(jobs):06d}',suite=suite,method=method,seed=int(seed),truth=list(truth),horizon=horizon,delta=config['delta'],**spec))
    b=config['binary']
    for k in b['kappas']:
        for truth in (0,1):
            for seed in b['seeds']:
                for method in b['methods']:
                    add('binary',method,seed,[truth],b['horizon'],kappa=k)
    c=config['recursive']
    for kappas in itertools.product(c['kappas'],repeat=2):
        for truth in itertools.product((0,1),repeat=2):
            for seed in c['seeds']:
                for method in c['methods']:
                    add('recursive',method,seed,truth,c['horizon'],kappas=list(kappas))
    c=config['nuisance']
    for m in c['dimensions']:
        for seed in c['seeds']:
            truth=np.random.default_rng(np.random.SeedSequence([seed,83])).integers(0,2,size=34).tolist()[:m+2]
            for method in c['methods']:
                add('nuisance',method,seed,truth,c['horizon'],m=m,kappa=c['kappa'])
    c=config['random']
    for i in range(c['trees']):
        rng=np.random.default_rng(np.random.SeedSequence([c['tree_seed'],i]))
        # Uniform leaf-label permutation on a complete depth-three binary tree.
        leaves=rng.permutation(8).tolist(); nodes=[]
        for depth in range(3):
            for prefix in itertools.product((0,1),repeat=depth):
                nodes.append(dict(name='p'+''.join(map(str,prefix)), prefix=list(prefix),
                     kappa=float(rng.uniform(*c['kappa_range'])), duration=int(rng.integers(1,6))))
        tree=dict(tree_id=i,nodes=nodes,leaf_labels=leaves)
        for seed in c['seeds']:
            leaf=int(np.random.default_rng(np.random.SeedSequence([seed,i,84])).integers(8))
            index=leaves.index(leaf); truth=[(index >> shift)&1 for shift in (2,1,0)]
            for method in c['methods']:
                add('random',method,seed,truth,c['horizon'],tree=tree,model_label=leaf)
    c=config['uav']
    for idx,_ in enumerate(c['theta']):
        for seed in c['seeds']:
            for method in c['methods']:
                add('uav',method,seed,[],c['horizon'],truth_index=idx)
    # Predeclared sensitivity: same binary instances, only SafeRefine; .05 already primary.
    primary=[dict(j) for j in jobs if j['suite']=='binary' and j['method']=='SafeRefine']
    for delta in config['delta_sensitivity']:
        if delta==config['delta']: continue
        for j in primary:
            j.update(job_id=f'j{len(jobs):06d}',delta=delta,sensitivity=True)
            jobs.append(dict(j))
    return jobs
