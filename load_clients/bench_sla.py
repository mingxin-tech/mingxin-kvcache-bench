import urllib.request, json, time, sys
from concurrent.futures import ThreadPoolExecutor

# modes:
#   populate <reps> <off> <N>              -> prefill N sessions (max_tokens=1) across 8 ports, store KV to LMCache
#   measure  <reps> <off> <N> <decode>     -> N concurrent users across 8 ports, report per-user TTFT & decode tps
#   calib    <reps>                         -> single request, print prompt_tokens
mode = sys.argv[1]
reps = int(sys.argv[2])
NPORTS = 8
PORTS = [8000 + i for i in range(NPORTS)]
basep = ('背景知识：AISSD5000是国产高性能全闪NVMe-oF存储，可作为大模型推理KV缓存的分层后备介质，'
         '配合vLLM与LMCache在显存/内存/磁盘之间分层存取KV，从而在高并发在线服务下降低首token时延并稳定输出速率。')

def make_prefix(i):
    return '[sess-%06d] ' % i + basep * reps

def call(port, sid, maxtok, stream):
    body = json.dumps({
        'model': 'qwen', 'stream': stream,
        'messages': [{'role': 'system', 'content': make_prefix(sid)},
                     {'role': 'user', 'content': '请简要回答第%d号问题。' % sid}],
        'max_tokens': maxtok, 'temperature': 0
    }).encode()
    url = 'http://127.0.0.1:%d/v1/chat/completions' % port
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800)

def pct(a, q):
    if not a: return 0.0
    return a[min(len(a) - 1, int(len(a) * q))]

if mode == 'calib':
    r = call(PORTS[0], 999999, 8, False)
    d = json.load(r)
    print('prompt_tokens=%d (reps=%d)' % (d['usage']['prompt_tokens'], reps), flush=True)
    sys.exit(0)

off = int(sys.argv[3])
N = int(sys.argv[4])

if mode == 'populate':
    def pop(idx):
        try:
            r = call(PORTS[idx % NPORTS], off + idx, 1, True)
            for _ in r: pass
            return 1
        except Exception:
            return 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=NPORTS * 2) as ex:
        ok = sum(ex.map(pop, range(N)))
    print('[populate] N=%d off=%d ok=%d wall=%.1fs' % (N, off, ok, time.time() - t0), flush=True)
    sys.exit(0)

# measure
decode = int(sys.argv[5]) if len(sys.argv) > 5 else 512

def user(idx):
    port = PORTS[idx % NPORTS]
    sid = off + idx
    st = time.time(); ttft = None; n = 0
    try:
        r = call(port, sid, decode, True)
        for line in r:
            sx = line.decode('utf-8', 'ignore')
            if sx.startswith('data:') and '"content"' in sx:
                if ttft is None: ttft = time.time() - st
                n += 1
        end = time.time()
        if ttft is None or n < 2:
            return None
        gen = end - st - ttft          # time spent producing tokens after the first
        dtps = (n - 1) / gen if gen > 1e-6 else 0.0
        return (ttft, dtps, n, end - st)
    except Exception:
        return None

t0 = time.time()
with ThreadPoolExecutor(max_workers=N) as ex:
    res = [x for x in ex.map(user, range(N)) if x is not None]
wall = time.time() - t0
if not res:
    print('[measure] N=%d ALL-FAIL' % N, flush=True); sys.exit(0)
tt = sorted(x[0] for x in res)
dt = sorted(x[1] for x in res)
tot = sum(x[2] for x in res)
mean_ttft = sum(tt) / len(tt)
mean_dtps = sum(dt) / len(dt)
ttft_p95 = pct(tt, 0.95)
ok_ttft = ttft_p95 < 5.0
ok_dtps = mean_dtps >= 30.0
verdict = 'PASS' if (ok_ttft and ok_dtps) else 'FAIL'
print(('[measure] N=%d ok=%d wall=%.1fs | TTFT p50=%.2f p95=%.2f p99=%.2f mean=%.2f s | '
       'decode/user p50=%.1f p95=%.1f mean=%.1f tok/s | agg=%.0f tok/s | TTFT_p95<5:%s decode>=30:%s => %s')
      % (N, len(res), wall, pct(tt, .5), ttft_p95, pct(tt, .99), mean_ttft,
         pct(dt, .5), pct(dt, .95), mean_dtps, tot / wall,
         ok_ttft, ok_dtps, verdict), flush=True)
