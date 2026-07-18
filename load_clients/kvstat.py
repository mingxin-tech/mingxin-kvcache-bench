import re,statistics
for tag,fn in [("WS","/mnt/ws5000/kv480_ws.log"),("LOCAL","/mnt/ws5000/kv480_loc.log")]:
    tp=[];cost=[];sz=0.0
    for line in open(fn,'rb'):
        l=line.decode('utf-8','ignore')
        m=re.search(r'size:\s*([\d.]+)\s*gb, cost\s*([\d.]+)\s*ms, throughput:\s*([\d.]+)\s*GB/s',l)
        if m:
            sz+=float(m.group(1));cost.append(float(m.group(2)));tp.append(float(m.group(3)))
    if not tp:
        print(tag,"none");continue
    tp.sort();cost.sort()
    p=lambda a,q:a[min(len(a)-1,int(len(a)*q))]
    print("%s | n=%d cumGB=%.1f | tp GB/s mean=%.2f p50=%.2f p90=%.2f max=%.2f min=%.2f | cost ms mean=%.0f p50=%.0f p90=%.0f max=%.0f"%(
        tag,len(tp),sz,statistics.mean(tp),p(tp,.5),p(tp,.9),tp[-1],tp[0],statistics.mean(cost),p(cost,.5),p(cost,.9),cost[-1]))
