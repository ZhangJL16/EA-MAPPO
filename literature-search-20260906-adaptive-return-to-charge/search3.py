import json, time, urllib.request, urllib.parse, re, os
OUT = "literature-search-20260906-adaptive-return-to-charge/raw"
def get(url, tries=5, base=6.0):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"lit-survey/0.3 (mailto:research@example.org)"})
            with urllib.request.urlopen(req, timeout=45) as r: return r.read().decode("utf-8","replace")
        except Exception as e:
            if i==tries-1: return "__ERR__ "+str(e)
            time.sleep(base*(i+1))
def arxiv(q, n=12):
    return get("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q) + f"&start=0&max_results={n}")
def arx_hits(xml):
    if xml.startswith("__ERR__"): return [("ERR", xml[:90],"?")]
    out=[]
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t=re.search(r"<title>(.*?)</title>",e,re.S); y=re.search(r"<published>(\d{4})",e); i=re.search(r"<id>(http://arxiv.org/abs/[^<]+)</id>",e)
        out.append((t.group(1).strip().replace("\n"," ") if t else "?", y.group(1) if y else "?", i.group(1) if i else "?"))
    return out
ARXIV_Q = {
 "B1_chance_energy_planning": 'abs:"chance-constrained" AND abs:"energy" AND abs:"planning"',
 "B2_risk_aware_battery": 'abs:"risk-aware" AND abs:"battery"',
 "B3_optimal_stopping_battery": 'abs:"optimal stopping" AND abs:"battery"',
 "B4_mission_abort_policy": 'abs:"mission abort"',
 "B5_safe_return_uav": 'abs:"safe return" AND abs:"UAV"',
 "B6_remaining_flight_time": 'abs:"remaining flight time"',
 "B7_battery_aware_rl": 'abs:"battery-aware" AND abs:"reinforcement learning"',
 "B8_energy_aware_navigation_rl": 'abs:"energy-aware" AND abs:"navigation" AND abs:"reinforcement learning"',
 "B9_reach_avoid_battery": 'abs:"reach-avoid" AND abs:"battery"',
 "B10_return_threshold_energy": 'abs:"return" AND abs:"energy threshold" AND abs:"robot"',
}
lines=[]
for tag,q in ARXIV_Q.items():
    a=arxiv(q); time.sleep(3.3)
    json.dump({"query":q,"arxiv":a}, open(f"{OUT}/{tag}.json","w"))
    lines.append(f"\n=== {tag}: {q}\n" + "\n".join(f"  {t} | {y} | {i}" for t,y,i in arx_hits(a)))
open(f"{OUT}/_digest3.txt","w").write("\n".join(lines))
print("\n".join(lines)[:6500])
