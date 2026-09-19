"""Exact finite-tree renewal-ratio DP. No simulator state abstraction."""
from __future__ import annotations


def solve(nodes, roots, candidate=None, tolerance=1e-11):
    """Nodes have tau_R, safe_C, children [{id,time,probability}].

    Every represented D state must admit R. Unknown successors are an error,
    never implicit stopping leaves. The optional candidate has zero root weight.
    """
    visiting=set(); checked=set()
    def check(key):
        if key in visiting: raise ValueError('cycle in task-sequence tree')
        if key in checked:return
        if key not in nodes:raise ValueError(f'incomplete tree: {key}')
        visiting.add(key); n=nodes[key]
        if not n['R_safe'] or n['tau_R']<=0:raise ValueError('R not feasible')
        if n['safe_C']:
            if len(n['children'])!=3 or any(abs(c['probability']-1/3)>1e-15 or c['time']<=0 for c in n['children']):
                raise ValueError('expected three uniform positive-duration branches')
            for c in n['children']:check(c['id'])
        visiting.remove(key);checked.add(key)
    if len(roots)!=3 or any(abs(r['probability']-1/3)>1e-15 or r['time']<=0 for r in roots):
        raise ValueError('expected three mandatory H task branches')
    for root in roots:check(root['id'])
    if candidate:check(candidate)

    def evaluate(rho, vb=False):
        values={};decisions={};advantages={}
        def visit(key):
            if key in values:return values[key]
            n=nodes[key];chosen=(0.,n['tau_R']);action='R';delta=None
            if n['safe_C']:
                outcomes=[(c,visit(c['id'])) for c in n['children']]
                cn=sum(c['probability']*(1+v[0]) for c,v in outcomes)
                ct=sum(c['probability']*(c['time']+v[1]) for c,v in outcomes)
                delta=cn-rho*ct+rho*n['tau_R']
                if vb or delta>0:chosen=(cn,ct);action='C'
            values[key]=chosen;decisions[key]=action;advantages[key]=delta
            return chosen
        outcomes=[(r,visit(r['id'])) for r in roots]
        count=sum(r['probability']*(1+v[0]) for r,v in outcomes)
        duration=sum(r['probability']*(r['time']+v[1]) for r,v in outcomes)
        if candidate:visit(candidate)
        return count,duration,decisions,advantages

    rho=0.;history=[]
    for iteration in range(100):
        count,duration,_,_=evaluate(rho)
        residual=count-rho*duration
        history.append(dict(iteration=iteration,rho=rho,tasks=count,time=duration,excess=residual))
        updated=count/duration
        if abs(residual)<=tolerance:
            rho=updated;break
        rho=updated
    else:raise RuntimeError('Dinkelbach failed to converge')
    count,duration,optimal,delta=evaluate(rho)
    bn,bt,vb,_=evaluate(rho,True)
    low=max(0.,rho-1e-11);high=rho+1e-11
    ln,lt,_,_=evaluate(low);hn,ht,_,_=evaluate(high)
    if ln-low*lt < -tolerance or hn-high*ht > tolerance:raise RuntimeError('ratio root certificate fails')
    def occupancy(policy):
        mass={}
        def visit(key,p):
            mass[key]=mass.get(key,0.)+p
            if policy[key]=='C':
                for c in nodes[key]['children']:visit(c['id'],p*c['probability'])
        for r in roots:visit(r['id'],r['probability'])
        return mass
    op=occupancy(optimal);bp=occupancy(vb)
    early=[dict(node=k,delta=delta[k],optimal_cycle_visit_mass=op.get(k,0.),
                vb_cycle_visit_mass=bp.get(k,0.)) for k in delta
           if nodes[k]['safe_C'] and delta[k]<-1e-9]
    ratio=(bn/bt)/rho
    if ratio>1+1e-9:raise RuntimeError('baseline exceeds optimum')
    # The bracket allows a threshold decision only when floating-point uncertainty
    # cannot straddle .98. Both policies use the same exact admissible-action tree.
    ratio_interval=[(bn/bt)/high,(bn/bt)/low] if low else [0.,float('inf')]
    decision='KILL' if ratio_interval[0]>=.98 else ('GAP_BELOW_98' if ratio_interval[1]<.98 else 'NUMERICALLY_UNRESOLVED')
    return dict(rho_star=rho,rho_VB=bn/bt,VB_over_oracle=ratio,ratio_interval=ratio_interval,
        decision=decision,oracle_tasks_per_cycle=count,oracle_time_per_cycle=duration,
        VB_tasks_per_cycle=bn,VB_time_per_cycle=bt,root_excess=count-rho*duration,
        rho_bracket=[low,high],bracket_excess=[ln-low*lt,hn-high*ht],iterations=history,
        actions=optimal,VB_actions=vb,delta_Q=delta,optimal_occupancy=op,VB_occupancy=bp,
        safe_return_optimal_nodes=early,candidate=(dict(node=candidate,action=optimal[candidate],
            safe_C=nodes[candidate]['safe_C'],delta_Q=delta[candidate],
            candidate_only=candidate not in op) if candidate else None),
        scope='exact ratio solution of fully enumerated frozen deterministic task-sequence tree under declared robust immediate-return safe set')
