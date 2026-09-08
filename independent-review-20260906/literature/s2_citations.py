import json, urllib.request, time, sys, os
def get(url, retries=8):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'lit-review/0.1'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i==retries-1: raise
            time.sleep(12*(i+1))
ids = {
 'back-to-base':'arXiv:2501.02620',
 'recovery-rl':'arXiv:2010.15920',
 'sailr':'arXiv:2106.06642',
 'rcrl':'arXiv:2202.01260',
 'leave-no-trace':'arXiv:1711.06782',
 'reach-avoid-rl':'arXiv:2105.07717',
 'notomista-persistification':'arXiv:1803.08305',
}
os.makedirs('literature/citations', exist_ok=True)
for k,pid in ids.items():
    out=f'literature/citations/{k}.json'
    if os.path.exists(out): continue
    try:
        d = get(f'https://api.semanticscholar.org/graph/v1/paper/{pid}/citations?fields=title,year,venue,abstract&limit=40')
        with open(out,'w') as f: json.dump(d,f,indent=1)
        print(k, '->', len(d.get('data',[])), 'citations saved')
    except Exception as e:
        print(k, 'ERR', e)
    time.sleep(20)
