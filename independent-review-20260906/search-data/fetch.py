import json, time, urllib.request, urllib.parse, os, sys

OUT = os.path.dirname(os.path.abspath(__file__))
def get(url, timeout=45):
    req = urllib.request.Request(url, headers={'User-Agent':'lit-review/0.1 (mailto:research@example.com)'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode('utf-8', 'replace')

def save(name, text):
    p = os.path.join(OUT, name)
    with open(p, 'w') as f: f.write(text)
    return p

# arXiv API queries
arx_queries = {
 'arxiv_nav_success.json': 'search_query=all:%22success+rate%22+AND+all:%22obstacle+avoidance%22+AND+all:%22reinforcement+learning%22+AND+cat:cs.RO&start=0&max_results=25',
 'arxiv_cbf_shield.json': 'search_query=all:%22safety+filter%22+AND+all:%22reinforcement+learning%22+AND+cat:cs.RO&start=0&max_results=25',
 'arxiv_safe_nav_2024.json': 'search_query=all:%22safe+navigation%22+AND+cat:cs.RO&start=0&max_results=25&sortBy=submittedDate&sortOrder=descending',
 'arxiv_safety_gymnasium.json': 'search_query=all:%22safety-gymnasium%22&start=0&max_results=20',
 'arxiv_focops.json': 'search_query=all:%22first-order+constrained+optimization%22+AND+all:%22policy+space%22&start=0&max_results=10',
}
for name, q in arx_queries.items():
    try:
        save(name, get('http://export.arxiv.org/api/query?'+q))
        print('OK', name)
    except Exception as e:
        print('FAIL', name, e)
    time.sleep(2)
