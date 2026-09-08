import json, re, glob, os

def arxiv_hits(xml):
    if xml.startswith("__ERR__"): return [("ERR",xml[:60],"?")]
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    out=[]
    for e in entries:
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        y = re.search(r"<published>(\d{4})", e)
        i = re.search(r"<id>(http://arxiv.org/abs/[^<]+)</id>", e)
        out.append((t.group(1).strip().replace("\n"," ") if t else "?", y.group(1) if y else "?", i.group(1) if i else "?"))
    return out

def s2_hits(j):
    if isinstance(j,str): return [("ERR",j[:60],"?")]
    out=[]
    for p in j.get("data", []):
        ext = p.get("externalIds") or {}
        out.append((p.get("title"), p.get("year"), p.get("venue"), p.get("citationCount"), ext.get("ArXiv"), p.get("paperId")))
    return out

def oa_hits(j):
    if isinstance(j,str): return [("ERR",j[:60],"?")]
    out=[]
    for w in j.get("results", []):
        pl = (w.get("primary_location") or {}).get("source") or {}
        out.append((w.get("title"), w.get("publication_year"), pl.get("display_name"), w.get("cited_by_count"), (w.get("doi") or "")))
    return out

lines=[]
for f in sorted(glob.glob("literature-search-20260906-adaptive-return-to-charge/raw/S*.json")):
    rec = json.load(open(f))
    tag = os.path.basename(f)[:-5]
    lines.append(f"\n=== {tag}: {rec['query']}")
    lines.append("-- arxiv --")
    for t,y,i in arxiv_hits(rec["arxiv"])[:8]: lines.append(f"  {t} | {y} | {i}")
    lines.append("-- s2 --")
    try: sj=json.loads(rec["s2"])
    except: sj=rec["s2"]
    for t,y,v,c,ax,pid in s2_hits(sj)[:8]: lines.append(f"  {t} | {y} | {v} | cit:{c} | {ax}")
    lines.append("-- openalex --")
    try: oj=json.loads(rec["openalex"])
    except: oj=rec["openalex"]
    for t,y,v,c,d in oa_hits(oj)[:8]: lines.append(f"  {t} | {y} | {v} | cit:{c} | {d}")

open("literature-search-20260906-adaptive-return-to-charge/raw/_digest.txt","w").write("\n".join(lines))
print("\n".join(lines)[:9000])
