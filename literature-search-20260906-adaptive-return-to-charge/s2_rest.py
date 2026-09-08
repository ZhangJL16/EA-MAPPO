import json, time, urllib.request, os
OUT="literature-search-20260906-adaptive-return-to-charge/raw"
def get(url, tries=8):
    for i in range(tries):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"lit-survey/1.5"})
            with urllib.request.urlopen(req, timeout=50) as r: return r.read().decode("utf-8","replace")
        except Exception as e:
            if i==tries-1: return "__ERR__ "+str(e)
            time.sleep(20*(i+1))
KEYS=[("rcppo","arXiv:2410.22600"),("learn2recharge","arXiv:2309.03157"),("rapcpo","arXiv:2605.11975"),("task_persist_icra22","DOI:10.1109/ICRA46639.2022.9812208")]
for key,pid in KEYS:
    mf=f"{OUT}/s2_{key}_meta.json"; cf=f"{OUT}/s2_{key}_citations.json"
    if not os.path.exists(mf):
        m=get("https://api.semanticscholar.org/graph/v1/paper/"+pid+"?fields=title,year,venue,citationCount,abstract,externalIds,authors")
        open(mf,"w").write(m); time.sleep(22)
    if not os.path.exists(cf):
        c=get("https://api.semanticscholar.org/graph/v1/paper/"+pid+"/citations?fields=title,year,venue,contexts,externalIds&limit=100")
        open(cf,"w").write(c); time.sleep(22)
    print(key,"done",flush=True)
print("S2_REST_DONE")
