import urllib.request,json,time,sys
from concurrent.futures import ThreadPoolExecutor
mode=sys.argv[1]; reps=int(sys.argv[2])
basep=('背景知识：AISSD5000是国产高性能全闪NVMe-oF存储，可作为大模型推理KV缓存的分层后备介质，'
       '配合vLLM与LMCache在显存/内存/磁盘之间分层存取KV，在高并发在线服务下降低首token时延并稳定输出速率。')
def mkp(i): return '[sess-%06d] '%i + basep*reps
def req(sid,maxtok):
    body=json.dumps({'model':'q3c','stream':True,'messages':[{'role':'system','content':mkp(sid)},{'role':'user','content':'请简答第%d题'%sid}],'max_tokens':maxtok,'temperature':0}).encode()
    st=time.time();ttft=None;n=0
    try:
        r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=body,headers={'Content-Type':'application/json'}),timeout=1800)
        for line in r:
            sx=line.decode('utf-8','ignore')
            if sx.startswith('data:') and '"content"' in sx:
                if ttft is None: ttft=time.time()-st
                n+=1
        return (ttft,time.time()-st,n)
    except Exception:
        return (None,None,0)
p=lambda a,q:a[min(len(a)-1,int(len(a)*q))] if a else 0.0
if mode=='populate':
    off=int(sys.argv[3]);N=int(sys.argv[4]);t0=time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        ok=sum(1 for x in ex.map(lambda i:req(off+i,1),range(N)) if x[0] is not None)
    print('populate N=%d ok=%d wall=%.1fs'%(N,ok,time.time()-t0));sys.exit(0)
off=int(sys.argv[3]);N=int(sys.argv[4]);conc=int(sys.argv[5]);dec=int(sys.argv[6]) if len(sys.argv)>6 else 8
t0=time.time()
with ThreadPoolExecutor(max_workers=conc) as ex:
    res=[x for x in ex.map(lambda i:req(off+i,dec),range(N)) if x[0] is not None]
wall=time.time()-t0
tt=sorted(r[0] for r in res)
print('conc=%d N=%d ok=%d wall=%.1fs TTFT_p50=%.2f p90=%.2f mean=%.2f'%(conc,N,len(res),wall,p(tt,.5),p(tt,.9),(sum(tt)/len(tt) if tt else 0)))