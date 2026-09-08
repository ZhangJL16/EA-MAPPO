import json, time, urllib.request, os
OUT = "literature-search-20260906-adaptive-return-to-charge/raw"
def get(url, tries=6):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"lit-survey/0.4 (mailto:research@example.org)"})
            with urllib.request.urlopen(req, timeout=50) as r: return r.read().decode("utf-8","replace")
        except Exception as e:
            if i==tries-1: return "__ERR__ "+str(e)
            time.sleep(18*(i+1))
KEYS = [
 ("btb","arXiv:2501.02620"),
 ("persist_ral","DOI:10.1109/LRA.2018.2789848"),
 ("persist_tcst","DOI:10.1109/TCST.2020.2978913"),
 ("energy_autonomy_tro","DOI:10.1109/TRO.2022.3175438"),
 ("esu_cbf","arXiv:2306.15115"),
 ("es_cbf_auro","DOI:10.1007/s10514-025-10203-w"),
 ("rcppo","arXiv:2410.22600"),
 ("learn2recharge","arXiv:2309.03157"),
 ("rapcpo","arXiv:2605.11975"),
 ("task_persist_icra22","DOI:10.1109/ICRA46639.2022.9812208"),
]
for key, pid in KEYS:
    mf=f"{OUT}/s2_{key}_meta.json"; cf=f"{OUT}/s2_{key}_citations.json"
    if not os.path.exists(mf):
        m=get("https://api.semanticscholar.org/graph/v1/paper/"+pid+"?fields=title,year,venue,citationCount,abstract,externalIds")
        open(mf,"w").write(m); time.sleep(16)
    if not os.path.exists(cf):
        c=get("https://api.semanticscholar.org/graph/v1/paper/"+pid+"/citations?fields=title,year,venue,externalIds,contexts&limit=100")
        open(cf,"w").write(c); time.sleep(16)
    print(key, "meta:", mf, "cit:", cf, flush=True)
print("S2DONE")
