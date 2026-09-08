import json, time, urllib.request, urllib.parse, re, os
OUT="literature-search-20260906-adaptive-return-to-charge/raw/fulltext"
def get(url, tries=4):
    for i in range(tries):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) lit-survey/2.1"})
            with urllib.request.urlopen(req, timeout=50) as r: return r.read().decode("utf-8","replace")
        except Exception as e:
            if i==tries-1: return "__ERR__ "+str(e)
            time.sleep(10*(i+1))
TITLES = {
 "eclares": "Eclares Energy-Aware Clarity-Driven Ergodic Search",
 "mesch": "meSch Multi-Agent Energy-Aware Scheduling for Task Persistence",
 "energy_tank": "A Novel Safety-Aware Energy Tank Formulation Based on Control Barrier Functions",
 "es_cbf": "ES-CBF energy sufficiency extension sample based path planners",
 "humanoid_safe_stop": "Humanoid Safe Stop via Learned Stoppability Value",
 "robust_reach_drl": "Robust reachability within deep reinforcement learning framework",
 "lr_bench": "Learning Reach-Avoid Task with Reinforcement Learning Vectorized Simulation and Benchmarks",
 "robust_peakcost": "Robust Peak-cost Constrained Reinforcement Learning",
 "epigraph_marl": "Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form",
 "vf_tl": "Value Functions for Temporal Logic Optimal Policies and Safety Filters",
}
res={}
for key,t in TITLES.items():
    f=f"{OUT}/arxiv_search_{key}.json"
    if os.path.exists(f): continue
    q=f"ti:{urllib.parse.quote(t)}"
    a=get("http://export.arxiv.org/api/query?search_query="+q+"&start=0&max_results=6")
    json.dump({"query":t,"resp":a}, open(f,"w"))
    hits=[]
    for e in re.findall(r"<entry>(.*?)</entry>", a, re.S):
        ti=re.search(r"<title>(.*?)</title>",e,re.S)
        i=re.search(r"<id>(http://arxiv.org/abs/[^<]+)</id>",e)
        if ti and i: hits.append((ti.group(1).strip(), i.group(1)))
    res[key]=hits
    print(key, "->", hits[:3], flush=True)
    time.sleep(14)
json.dump(res, open(f"{OUT}/_arxiv_title_hits.json","w"), indent=1)
print("DONE")
