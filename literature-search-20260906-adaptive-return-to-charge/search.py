import json, time, urllib.request, urllib.parse, sys, os

OUT = "literature-search-20260906-adaptive-return-to-charge/raw"
os.makedirs(OUT, exist_ok=True)

def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"lit-search/0.1 (research)"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if i == tries-1: return f"__ERR__ {e}"
            time.sleep(2+i*2)

def arxiv(q, n=15):
    url = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(f'all:"{q}"') + f"&start=0&max_results={n}"
    return get(url)

def s2_search(q, n=15):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?query=" + urllib.parse.quote(q) + f"&limit={n}&fields=title,year,venue,externalIds,citationCount,abstract"
    return get(url)

def openalex(q, n=15):
    url = "https://api.openalex.org/works?search=" + urllib.parse.quote(q) + f"&per-page={n}&select=id,doi,title,publication_year,primary_location,cited_by_count"
    return get(url)

QUERIES = {
 "S1_energy_rl_persistent": [
   "energy-aware reinforcement learning unmanned aerial vehicle battery charging",
   "persistent monitoring energy constraints recharging reinforcement learning",
   "energy constrained coverage path planning reinforcement learning UAV",
 ],
 "S2_return_decision": [
   "return to home decision policy battery drone",
   "abort and return mission energy optimal stopping robot",
   "when to return charging station reinforcement learning drone",
 ],
 "S3_energy_cbf": [
   "persistification of robotic tasks",
   "energy sufficiency control barrier functions unknown environments",
   "minimum energy control barrier function battery",
 ],
 "S4_back_to_base": [
   "Back to Base safe resets reach-avoid safety filters",
   "reach avoid return to recharge reinforcement learning",
 ],
 "S5_energy_to_go": [
   "energy to go prediction autonomous robot",
   "remaining battery prediction unmanned aerial vehicle machine learning return",
   "probability of safe return battery estimation drone",
 ],
 "S6_risk_aware": [
   "risk aware motion planning CVaR energy constraint UAV",
   "chance constrained path planning energy budget robot",
   "distributionally robust risk aware energy navigation",
 ],
 "S0_key_methods": [
   "Solving Minimum-Cost Reach Avoid using Reinforcement Learning",
   "Stochastic Minimum-Cost Reach-Avoid Reinforcement Learning",
   "Reachability Constrained Reinforcement Learning largest feasible set",
 ],
}

def arxiv_hits(xml):
    import re
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    out=[]
    for e in entries:
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        y = re.search(r"<published>(\d{4})", e)
        i = re.search(r"<id>(http://arxiv.org/abs/[^<]+)</id>", e)
        out.append((t.group(1).strip().replace("\n"," ") if t else "?", y.group(1) if y else "?", i.group(1) if i else "?"))
    return out

def s2_hits(j):
    out=[]
    for p in j.get("data", []):
        ext = p.get("externalIds") or {}
        out.append((p.get("title"), p.get("year"), p.get("venue"), p.get("citationCount"), ext.get("ArXiv"), p.get("paperId")))
    return out

def oa_hits(j):
    out=[]
    for w in j.get("results", []):
        pl = (w.get("primary_location") or {}).get("source") or {}
        out.append((w.get("title"), w.get("publication_year"), pl.get("display_name"), w.get("cited_by_count"), (w.get("doi") or "")))
    return out

summary = []
for group, qs in QUERIES.items():
    for qi, q in enumerate(qs):
        tag = f"{group}_{qi}"
        a = arxiv(q); time.sleep(3.2)
        s = s2_search(q); time.sleep(1.5)
        o = openalex(q); time.sleep(0.5)
        rec = {"group":group, "query":q, "arxiv":a, "s2":s, "openalex":o}
        with open(f"{OUT}/{tag}.json","w") as f: json.dump(rec,f)
        try: aj = json.loads(s)
        except Exception: aj = {}
        try: oj = json.loads(o)
        except Exception: oj = {}
        summary.append(f"\n=== {tag}: {q}\n-- arxiv --\n" + "\n".join(f"  {t} | {y} | {i}" for t,y,i in arxiv_hits(a)[:8]))
        summary.append("-- s2 --\n" + "\n".join(f"  {t} | {y} | {v} | cit:{c} | arXiv:{ax}" for t,y,v,c,ax,_ in s2_hits(aj)[:8]))
        summary.append("-- openalex --\n" + "\n".join(f"  {t} | {y} | {v} | cit:{c} | {d}" for t,y,v,c,d in oa_hits(oj)[:8]))

with open(f"{OUT}/_digest.txt","w") as f: f.write("\n".join(summary))
print("DONE", len(summary), "blocks")
