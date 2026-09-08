import json, time, urllib.request, urllib.parse, re, os
OUT = "literature-search-20260906-adaptive-return-to-charge/raw"
def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"lit-survey/0.2 (mailto:research@example.org)"})
            with urllib.request.urlopen(req, timeout=45) as r: return r.read().decode("utf-8","replace")
        except Exception as e:
            if i==tries-1: return "__ERR__ "+str(e)
            time.sleep(3*(i+1))
def arxiv(q, n=12):
    return get("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) + f"&start=0&max_results={n}")
def oa(q, n=12, filt="title_and_abstract.search"):
    return get(f"https://api.openalex.org/works?filter={filt}:{urllib.parse.quote(q)}&per-page={n}&select=id,doi,title,publication_year,primary_location,cited_by_count&sort=relevance_score:desc")

ARXIV_Q = {
 "A1_return_to_base": 'abs:"return-to-base" OR abs:"return to base"',
 "A2_abort_battery": 'abs:"abort" AND abs:"battery"',
 "A3_persistent_monitor_rl": 'abs:"persistent monitoring" AND abs:"reinforcement learning" AND abs:"energy"',
 "A4_recharging_learning": 'abs:"recharging" AND abs:"reinforcement learning" AND abs:"UAV"',
 "A5_energy_sufficiency_cbf": 'abs:"energy sufficiency" AND abs:"control barrier"',
 "A6_energy_to_go": 'abs:"energy-to-go" OR abs:"energy to go"',
 "A7_remaining_battery_return": 'abs:"remaining battery" AND abs:"return"',
 "A8_cvar_energy_path": 'abs:"CVaR" AND abs:"energy" AND abs:"planning"',
 "A9_chance_constrained_battery": 'abs:"chance-constrained" AND abs:"battery"',
 "A10_safe_return_rl": 'abs:"safe return" AND abs:"reinforcement learning"',
 "A11_reach_avoid_recharge": 'abs:"reach-avoid" AND abs:"recharge"',
 "A12_return_home_drone": 'abs:"return-to-home" OR abs:"return to home" AND abs:"drone"',
}
OA_Q = {
 "O1": "return to base energy drone decision",
 "O2": "mission abort policy energy optimal stopping",
 "O3": "energy-aware reinforcement learning persistent UAV monitoring charging",
 "O4": "safe return probability battery robot",
 "O5": "energy to go prediction autonomous aerial",
 "O6": "remaining flight time prediction drone battery machine learning",
 "O7": "chance-constrained path planning energy budget robot",
 "O8": "CVaR energy constrained trajectory planning UAV",
 "O9": "risk-aware return-to-home unmanned aerial vehicle",
 "O10": "recharge scheduling persistent coverage robot reinforcement learning",
 "O11": "energy sufficiency control barrier function",
 "O12": "reach-avoid safety filter reset hands-off learning",
 "O13": "battery-aware deep reinforcement learning navigation charging",
 "O14": "predictive safety network resource constrained",
}
def arx_hits(xml):
    if xml.startswith("__ERR__"): return [("ERR", xml[:80],"?")]
    out=[]
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t=re.search(r"<title>(.*?)</title>",e,re.S); y=re.search(r"<published>(\d{4})",e); i=re.search(r"<id>(http://arxiv.org/abs/[^<]+)</id>",e)
        out.append((t.group(1).strip().replace("\n"," ") if t else "?", y.group(1) if y else "?", i.group(1) if i else "?"))
    return out
def oa_hits(j):
    if isinstance(j,str): return [("ERR",j[:80],"","","")]
    out=[]
    for w in j.get("results",[]):
        pl=(w.get("primary_location") or {}).get("source") or {}
        out.append((w.get("title"), w.get("publication_year"), pl.get("display_name"), w.get("cited_by_count"), w.get("doi") or ""))
    return out
lines=[]
for tag,q in ARXIV_Q.items():
    a=arxiv(q); time.sleep(3.3)
    json.dump({"query":q,"arxiv":a}, open(f"{OUT}/{tag}.json","w"))
    lines.append(f"\n=== {tag}: {q}\n" + "\n".join(f"  {t} | {y} | {i}" for t,y,i in arx_hits(a)))
for tag,q in OA_Q.items():
    o=oa(q); time.sleep(0.6)
    json.dump({"query":q,"openalex":o}, open(f"{OUT}/{tag}.json","w"))
    lines.append(f"\n=== {tag}: {q}\n" + "\n".join(f"  {t} | {y} | {v} | cit:{c} | {d}" for t,y,v,c,d in oa_hits(o)))
open(f"{OUT}/_digest2.txt","w").write("\n".join(lines))
print("\n".join(lines)[:8000])
