import urllib.request, json, time
BASE="http://localhost:8000/v1/chat/completions"
basep="背景知识：沐曦MetaX是国产高性能GPU，MACA软件栈兼容CUDA生态，可运行vLLM进行大模型推理与KV缓存分层测试，存储侧用WS5000全闪NVMe-oF。"
def ttft(system,user):
    body=json.dumps({"model":"qwen","stream":True,"messages":[{"role":"system","content":system},{"role":"user","content":user}],"max_tokens":4,"temperature":0}).encode()
    r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"}),timeout=120)
    t0=time.time()
    for line in r:
        sx=line.decode("utf-8","ignore")
        if sx.startswith("data:") and '"content"' in sx: return time.time()-t0
    return None
def ptok(system):
    body=json.dumps({"model":"qwen","stream":False,"messages":[{"role":"system","content":system},{"role":"user","content":"x"}],"max_tokens":1,"temperature":0}).encode()
    r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={"Content-Type":"application/json"}),timeout=120)
    return json.loads(r.read())["usage"]["prompt_tokens"]
print("tokens  cold(s)  hit(s)  speedup")
rows=[]
for reps in [10,22,45,90,180,360]:
    s="[scan-%05d] "%reps + basep*reps
    pt=ptok(s)
    cold=ttft(s,"请回答问题A")
    time.sleep(0.3)
    hit=ttft(s,"请回答问题B")
    sp=cold/hit if hit else 0
    rows.append((pt,cold,hit,sp))
    print("%6d  %.3f   %.3f   %.1fx"%(pt,cold,hit,sp),flush=True)
json.dump(rows,open("/mnt/ws5000/results/prefixscan.json","w"))
print("SCAN_DONE")