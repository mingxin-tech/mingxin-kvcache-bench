import urllib.request, json, time, sys, os
from concurrent.futures import ThreadPoolExecutor
BASE="http://localhost:8000/v1/chat/completions"
MODEL="qwen"
label=sys.argv[1]; reps=int(sys.argv[2]); N=int(sys.argv[3]); mode=sys.argv[4]
decode=int(sys.argv[5]) if len(sys.argv)>5 else 512
conc=int(sys.argv[6]) if len(sys.argv)>6 else 16
basep="背景知识：AISSD5000是国产高性能全闪NVMe-oF存储，可作为大模型推理KV缓存的分层后备介质，配合vLLM与LMCache在显存/内存/磁盘之间分层存取KV。"
def make_prefix(i): return "[sess-%05d] "%i + basep*reps
def req(sid, maxtok):
    body=json.dumps({"model":MODEL,"stream":True,"messages":[{"role":"system","content":make_prefix(sid)},{"role":"user","content":"回答%d"%sid}],"max_tokens":maxtok,"temperature":0}).encode()
    st=time.time(); ttft=None; n=0
    try:
        r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"}),timeout=1200)
        for line in r:
            sx=line.decode("utf-8","ignore")
            if sx.startswith("data:") and '"content"' in sx:
                if ttft is None: ttft=time.time()-st
                n+=1
        return (ttft, time.time()-st, n)
    except Exception:
        return (None,None,0)
def pct(a,q): return a[min(len(a)-1,int(len(a)*q))] if a else 0.0
if mode=="populate":
    t0=time.time()
    for i in range(N): req(i,1)
    print("[%s] populate N=%d wall=%.1fs"%(label,N,time.time()-t0),flush=True); sys.exit(0)
res=[]; t0=time.time()
with ThreadPoolExecutor(max_workers=conc) as ex:
    futs=[ex.submit(req,i,decode) for i in range(N)]
    for f in futs:
        x=f.result()
        if x[0] is not None: res.append(x)
wall=time.time()-t0
tt=sorted(r[0] for r in res)
tpots=sorted((r[1]-r[0])/(r[2]-1) for r in res if r[2]>1)
tot=sum(r[2] for r in res)
summ={"label":label,"n":len(res),"decode":decode,"conc":conc,"wall_s":round(wall,1),
 "ttft_p50":round(pct(tt,.5),3),"ttft_p90":round(pct(tt,.9),3),"ttft_p99":round(pct(tt,.99),3),"ttft_mean":round(sum(tt)/len(tt),3) if tt else 0,
 "tpot_p50_ms":round(pct(tpots,.5)*1000,1),"req_s":round(len(res)/wall,2),"out_tok_s":round(tot/wall,1),"tot_out":tot}
print("[%s] n=%d wall=%.1fs TTFT p50=%.3f p90=%.3f p99=%.3f mean=%.3f | TPOT p50=%.1fms | req/s=%.2f out_tok/s=%.1f"%(label,summ["n"],wall,summ["ttft_p50"],summ["ttft_p90"],summ["ttft_p99"],summ["ttft_mean"],summ["tpot_p50_ms"],summ["req_s"],summ["out_tok_s"]),flush=True)
os.makedirs("/mnt/ws5000/results",exist_ok=True)
json.dump(summ,open("/mnt/ws5000/results/%s.json"%label,"w"),ensure_ascii=False,indent=2)
print("DONE",flush=True)