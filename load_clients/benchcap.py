import urllib.request, json, time, sys, statistics, os
BASE="http://localhost:8000/v1/chat/completions"
label=sys.argv[1]; reps=int(sys.argv[2]); N=int(sys.argv[3]); passes=int(sys.argv[4]) if len(sys.argv)>4 else 2
TH=float(sys.argv[5]) if len(sys.argv)>5 else 0.7
basep="背景知识：沐曦MetaX是国产高性能GPU，MACA软件栈兼容CUDA生态，可运行vLLM进行大模型推理与KV缓存分层测试，存储侧用WS5000全闪NVMe-oF。"
def make_prefix(i): return "[sess-%05d] "%i + basep*reps
def req(system,user,stream=True):
    body=json.dumps({"model":"qwen","stream":stream,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"max_tokens":4,"temperature":0}).encode()
    r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"}),timeout=600)
    if not stream: return json.loads(r.read())
    t0=time.time()
    for line in r:
        sx=line.decode("utf-8","ignore")
        if sx.startswith("data:") and '"content"' in sx: return time.time()-t0
    return None
u=req(make_prefix(0),"calib",stream=False); ptok=u["usage"]["prompt_tokens"]
res={"label":label,"reps":reps,"N":N,"prompt_tokens":ptok,"threshold":TH,"passes":{}}
def pct(a,q): return a[min(len(a)-1,int(len(a)*q))]
for p in range(1,passes+1):
    ts=[]; t0=time.time()
    for i in range(N):
        x=req(make_prefix(i),"第%d轮回答%d"%(p,i))
        if x is not None: ts.append(x)
    a=sorted(ts)
    rec=sum(1 for t in ts if t<TH); rcmp=sum(1 for t in ts if t>=TH)
    summ={"n":len(ts),"recovered":rec,"recompute":rcmp,"p50":round(statistics.median(a),4),"p90":round(pct(a,0.9),4),"p99":round(pct(a,0.99),4),"mean":round(sum(a)/len(a),4),"min":round(min(a),4),"max":round(max(a),4),"wall_s":round(time.time()-t0,1),"raw":[round(x,4) for x in ts]}
    res["passes"][p]=summ
    print("[%s] pass%d n=%d recovered=%d recompute=%d p50=%.3f p90=%.3f p99=%.3f mean=%.3f wall=%.1fs"%(label,p,summ["n"],rec,rcmp,summ["p50"],summ["p90"],summ["p99"],summ["mean"],summ["wall_s"]),flush=True)
os.makedirs("/mnt/ws5000/results",exist_ok=True)
json.dump(res,open("/mnt/ws5000/results/%s.json"%label,"w"),ensure_ascii=False,indent=2)
print("DONE prompt_tokens=%d"%ptok)