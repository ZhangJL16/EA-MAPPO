import json, time, urllib.request, urllib.parse, re, os
OUT="literature-search-20260906-adaptive-return-to-charge/raw"
def get(url, tries=5):
    for i in range(tries):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"lit-survey/1.2"})
            with urllib.request.urlopen(req, timeout=45) as r: return r.read().decode("utf-8","replace")
        except Exception as e:
            if i==tries-1: return "__ERR__ "+str(e)
            time.sleep(20*(i+1))
Q = {
 "B5_safe_return_uav": 'abs:"safe return" AND abs:"UAV"',
 "B6_remaining_flight_time": 'abs:"remaining flight time"',
 "B7_battery_aware_rl": 'abs:"battery-aware" AND abs:"reinforcement learning"',
 "B8_energy_aware_navigation_rl": 'abs:"energy-aware" AND abs:"navigation" AND abs:"reinforcement learning"',
 "B9_reach_avoid_battery": 'abs:"reach-avoid" AND abs:"battery"',
 "B10_return_threshold_energy": 'abs:"return" AND abs:"energy threshold" AND abs:"robot"',
 "B11_energy_budget_learning": 'abs:"energy budget" AND abs:"learning"',
 "B12_charging_decision_rl": 'abs:"charging decision" AND abs:"reinforcement learning"',
}
for tag,q in Q.items():
    f=f"{OUT}/{tag}.json"
    if os.path.exists(f) and not open(f).read().startswith('{"query"'):
        pass
    if os.path.exists(f) and "__ERR__" not in open(f).read(): continue
    a=get("http://export.arxiv.org/api/query?search_query="+urllib.parse.quote(q)+"&start=0&max_results=12")
    json.dump({"query":q,"arxiv":a}, open(f,"w"))
    print(tag, "saved", flush=True)
    time.sleep(18)
print("ARXIV_B_DONE")
