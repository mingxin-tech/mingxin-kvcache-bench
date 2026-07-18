#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
sudo -A bash -c 'cat > /tmp/agg.py' <<'PYEOF'
import re, glob, math
def pct(a,q):
    if not a: return 0.0
    return a[min(len(a)-1, max(0, math.ceil(q*len(a))-1))]
tags = ["FS_OWN8","FS_OWN16","FS_CROSS16","WS8","WS16","LOC8","LOC16","RC8","RC16"]
for tag in tags:
    tts=[]; tps=[]; walls=[]
    for side in "AB":
        fn = "/mnt/ws5000/results/m2_%s_%s.log" % (tag, side)
        try: txt = open(fn, errors="ignore").read()
        except: continue
        for m in re.finditer(r"REQ sid=\d+ ttft=([\d.]+) e2e=([\d.]+) ntok=(\d+) tpot_ms=([\d.]+)", txt):
            tts.append(float(m.group(1))); tps.append(float(m.group(4)))
        w = re.search(r"wall=([\d.]+)s", txt.split("REQ")[-1])
        wm = re.findall(r"\] n=\d+ wall=([\d.]+)s", txt)
        if wm: walls.append(float(wm[-1]))
    if not tts: print(tag, "NO DATA"); continue
    tts.sort(); tps.sort()
    n=len(tts); wall=max(walls) if walls else 0
    agg = n*64/wall if wall else 0
    print("%-11s n=%2d | TTFT p50=%6.2f p90=%6.2f p99=%6.2f mean=%6.2f max=%6.2f | TPOT p50=%7.1f p99=%7.1f mean=%7.1f | slower_wall=%6.1f agg_tok/s=%5.1f" %
          (tag, n, pct(tts,.5), pct(tts,.9), pct(tts,.99), sum(tts)/n, tts[-1], pct(tps,.5), pct(tps,.99), sum(tps)/len(tps), wall, agg))
PYEOF
sudo -A python3 /tmp/agg.py
