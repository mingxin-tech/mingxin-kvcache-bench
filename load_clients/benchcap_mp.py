import urllib.request, json, time, sys, os
from concurrent.futures import ThreadPoolExecutor
PORT=os.environ.get("BC_PORT","8000")
BASE="http://localhost:%s/v1/chat/completions"%PORT
MODEL="qwen"
label=sys.argv[1]; reps=int(sys.argv[2]); N=int(sys.argv[3]); mode=sys.argv[4]
decode=int(sys.argv[5]) if len(sys.argv)>5 else 512
conc=int(sys.argv[6]) if len(sys.argv)>6 else 16
basep="background: vLLM+LMCache tiers KV to external storage for KV readback bandwidth benchmark. "
def make_prefix(i): return "[sess-%05d] "%i + basep*reps
def req(sid, maxtok):
    body=json.dumps({"model":MODEL,"stream":True,"stream_options":{"include_usage":True},"messages":[{"role":"system","content":make_prefix(sid)},{"role":"user","content":"answer %d"%sid}],"max_tokens":maxtok,"temperature":0}).encode()
    st=time.time(); ttft=None; ctok=0
    try:
        r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"}),timeout=600)
        for line in r:
            sx=line.decode("utf-8","ignore")
            if not sx.startswith("data:"): continue
            if '"content"' in sx and ttft is None: ttft=time.time()-st
            if '"completion_tokens"' in sx:
                try:
                    u=json.loads(sx[5:].strip()).get("usage")
                    if u and u.get("completion_tokens"): ctok=u["completion_tokens"]
                except Exception: pass
        return (ttft, time.time()-st, ctok)
    except Exception:
        return (None,None,0)
def pct(a,q): return a[min(len(a)-1,int(len(a)*q))] if a else 0.0
if mode=="populate":
    t0=time.time()
    for i in range(N): req(i,1)
    print("[%s] populate N=%d wall=%.1fs"%(label,N,time.time()-t0)); sys.exit(0)
res=[]; t0=time.time()
with ThreadPoolExecutor(max_workers=conc) as ex:
    futs=[ex.submit(req,i,decode) for i in range(N)]
    for f in futs:
        x=f.result()
        if x[0] is not None: res.append(x)
wall=time.time()-t0
tt=sorted(r[0] for r in res); tot=sum(r[2] for r in res)
summ={"label":label,"n":len(res),"wall_s":round(wall,1),"ttft_p50":round(pct(tt,.5),3),"ttft_p99":round(pct(tt,.99),3),"gen_tok_per_req":round(tot/max(1,len(res)),1),"req_s":round(len(res)/wall,2),"out_tok_s":round(tot/wall,1)}
print("[%s] n=%d wall=%.1fs TTFT p50=%.3f p99=%.3f tok/req=%.1f req/s=%.2f out_tok/s=%.1f"%(label,summ["n"],wall,summ["ttft_p50"],summ["ttft_p99"],summ["gen_tok_per_req"],summ["req_s"],summ["out_tok_s"]))
os.makedirs("/mnt/ws5000/results",exist_ok=True)
json.dump(summ,open("/mnt/ws5000/results/%s.json"%label,"w"))
print("DONE")