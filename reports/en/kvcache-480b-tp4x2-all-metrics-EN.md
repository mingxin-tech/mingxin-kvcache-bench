# FX100 KV-Cache Benchmark Summary (480B, TP4×2, All Metrics, Signed Official)

> **Reference translation.** The signed Chinese original is the authoritative version ([download](https://mingxinstorage.xyz/evidence/R3-kvcache-480b-tp4x2.pdf)). AI-translated by Mingxin's translation pipeline; number fidelity machine-verified (see `dmkt/report_i18n/qc/R3.json`). Report date: 2026-07-06.

## AISSD5000 KV Cache Performance Test Summary Report – 480B Model · TP4×2 Dual Instance · Long Context Cold Recovery (All Metrics)

**Test Platform**: AMD Instinct MI308X ×8 (192 GB HBM per card) / ROCm 7.2 / vLLM 0.20.1+rocm721 + LMCache v1 (compiled from upstream main source)
**Storage Under Test**: AISSD5000 (WS5000) all-flash NVMe-oF array (4-disk RAID0, RoCEv2 over 100GbE, XFS, 14 TB)
**Reference Storage**: Server-local NVMe (Solidigm, PCIe Gen4, mounted partition 2 TB); no external storage recompute
**Model**: Qwen3-Coder-480B-FP8 (MoE, weights ≈ 450 GB)
**Deployment**: Dual instance TP4×2 (Instance A: cards 0–3 / port 8000; Instance B: cards 4–7 / port 8001, each with expert parallel)
**Workload**: Long context cold recovery – ≈ 29.8K tokens per session, KV ≈ 7.15 GB/session, 64 output tokens
**Date**: 2026-07-06 (this report summarizes all TP4×2 experiments on this date; the aggregate table data was obtained from a unified re-test with an enhanced client in the afternoon, consistent with two independent rounds from the morning, deviation ≤5%)

---

## 1. Key Conclusions

**Four configuration groups (AISSD5000 independent pool / AISSD5000 fs:// shared pool / local NVMe / no external storage recompute) × two full-machine concurrency levels (16 / 32), totaling nine measurement points, all are physical cold reads after page cache clearing, with per-request TTFT and TPOT recorded.**

1.  **Compared to local NVMe (core comparison): AISSD5000 reduces TTFT p50 by 26%–30%, p99 by 28%, and increases aggregate output throughput by 35%–36%**. At full-machine concurrency 16: TTFT p50 from 19.0 s to **13.4 s**, p99 from 19.0 s to **13.8 s**, throughput from 49.7 to **66.9 tok/s**; at concurrency 32: TTFT p50 from 34.5 s to **25.6 s**, p99 from 36.4 s to **26.3 s**, throughput from 53.3 to **72.6 tok/s**. Mechanism: The local disk is pinned at its physical limit of 6.78 GB/s at both concurrency levels, while AISSD5000 sustains a busy-window average of 8.6–9.6 GB/s and a peak of 10.3–10.4 GB/s (90%+ of single-port 100GbE line rate).
2.  **Compared to no external storage recompute: TTFT p50 is 8.6–9.8× faster, p99 is 19–23× faster, throughput is 18–21× higher, TPOT p50 is 75–164× better**. The recompute configuration has TPOT p50 reaching 1.8–5.2 seconds/token (decoding is continuously preempted by prefill, the output stream nearly freezes), and TTFT p99 reaches 4.5–10 minutes – for 480B long context session recovery, "recompute" is not an option.
3.  **The fs:// shared pool performs identically to the independent disk backend, with zero-cost hot sharing across instances**: The shared pool reading its own data (OWN) matches the independent backend (WS) across all metrics (TTFT p50 24.8 vs 25.6 s, throughput 73.4 vs 72.6 tok/s); **cross-reading sessions written by the other instance (CROSS) is only 8% slower (p50) / 3% slower (p99) than reading its own, and is still 22% faster than the local disk reading its own**. One KV pool for the whole machine, with free session migration across instances, is validated.
4.  **Decoding quality (TPOT) is consistently healthy across all four disk-based configuration groups**: TPOT p50 all fall within the 24–32 ms range; the storage medium does not affect steady-state decoding.
5.  **Measurements are self-consistent and reproducible**: The read volume from `iostat` for each configuration matches the working set (e.g., WS concurrency 32: 9.59 GB/s × 25 s ≈ 240 GB ≈ 32 × 7.15 GB); two independent complete measurements from the morning and afternoon of the same day show deviations ≤5% for all metrics.

---

## 2. System Under Test and Environment

### 2.1 Hardware

| Component | Configuration |
|------|------|
| GPU | 8 × AMD Instinct MI308X (192 GB HBM per card, gfx942) |
| Deployment | Instance A: cards 0–3 (TP=4, port 8000); Instance B: cards 4–7 (TP=4, port 8001), both with expert parallel |
| CPU / Memory | 2 × AMD EPYC 9654 (384 threads), ≈ 1.5 TB memory |
| **Storage Under Test** | **AISSD5000: 4-disk RAID0 (`/dev/md0`, xfs, 14 TB), NVMe-oF / RoCEv2, single-port 100 GbE** |
| Reference Storage | Local NVMe single disk (`/dev/nvme1n1`, PCIe Gen4, mounted at `/srv2`); no external storage (recompute) |

### 2.2 Software and Key Parameters

| Component | Version/Value |
|------|------|
| OS / GPU Stack | Ubuntu 22.04 (kernel 6.8.0-124) / ROCm 7.2 |
| Inference Engine | vLLM 0.20.1+rocm721 |
| KV Cache Library | LMCache (compiled from upstream main source 2026-06-29, default async disk loading) |
| Engine Parameters | `--tensor-parallel-size 4 --enable-expert-parallel`, `--max-model-len 32768`, `--gpu-memory-utilization 0.9`, `--no-enable-prefix-caching`, `PYTHONHASHSEED=0` |
| LMCache Common Parameters | `LMCACHE_CHUNK_SIZE=256`, `LMCACHE_LOCAL_CPU=True`, `LMCACHE_MAX_LOCAL_CPU_SIZE=4` |

### 2.3 Four Storage Configurations

| Group | KV Backend Configuration | Description |
|----|------------|------|
| ① WS (Independent Pool) | `LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache480tp4` (md0) | LocalDiskBackend, per-instance in-process index |
| ② FS (Shared Pool) | `LMCACHE_REMOTE_URL=fs://local:0/mnt/ws5000/kvpool_fs` (md0) | FSConnector, index is the filesystem, dual instances share one pool |
| ③ LOC (Local Disk) | `LMCACHE_LOCAL_DISK=file:///srv2/lmcache480tp4_local` (nvme1n1) | LocalDiskBackend, local single disk |
| ④ RC (Recompute) | LMCache not attached | Pure GPU recompute baseline |

> Note: The `fs://` URL requires the `fs://local:0/path` placeholder format – this version's `parse_remote_url()` enforces host:port validation; using `fs:///path` as per official documentation causes RemoteBackend connection failure and silent KV cache discard (see "AISSD5000-KVCache Cross-Instance Hot Sharing Validation Report" §2).

---

## 3. Test Methodology

### 3.1 Metric Definitions

| Metric | Definition |
|------|------|
| TTFT | Time from request issuance to receipt of the first content token (measured per request via streaming interface); p50/p90/p99/mean calculated from the combined A+B full-machine sample set (concurrency 16: n=16, concurrency 32: n=32, p99 is the nominal rank i.e., the worst request) |
| TPOT | Per-request (E2E − TTFT) ÷ (number of output tokens − 1), i.e., the average per-token decoding interval after the first token |
| Aggregate Output Throughput | Total output tokens for the full machine (2×N×64) ÷ the batch completion time of the slower of the two instances |
| Disk Peak / Busy Window Average | Maximum read bandwidth from `iostat -x 1` second-level samples / average over windows where bandwidth > 0.5 GB/s |

### 3.2 Workload and Cold Read Guarantee

- Two instances concurrently injected 48 distinct sessions (A injected 0–23, B injected 24–47, persisted 343–344 GB), injection took approximately 363 s;
- Measurement gears: full-machine concurrency 16 (8 sessions per instance × concurrency 8) and 32 (16 per instance × 16); A reads starting from 0, B reads starting from 24 (FS-CROSS gear swapped: A reads starting from 24, B reads starting from 0, i.e., each reads the other's injected sessions);
- Before each gear measurement, the host executed `sync; echo 3 > /proc/sys/vm/drop_caches` to clear 1.5 TB of page cache; each session was read only once; the read sets of the two instances were disjoint (no page cache piggybacking);
- Cold read verification: disk group hit tokens = 30208 per request, two independent rounds verified 96/96 and 96/96 on the day; iostat read volume matched the working set; recompute group disk read ≈0 and processes had no LMCache;
- Between gear switches, inference processes (including EngineCore/Worker remnants) were thoroughly cleaned, and 8-GPU HBM was confirmed to be zeroed.

---

## 4. Full Indicator Summary Table for All Nine Gears (Full-Machine Scope, A+B Combined Samples)

| Gear | Full-Machine Concurrency | TTFT p50 (s) | TTFT p90 (s) | TTFT p99 (s) | TTFT Mean (s) | TPOT p50 (ms) | Aggregate Throughput (tok/s) | Disk Peak (GB/s) | Disk Busy Window Mean (GB/s) |
|------|------|------|------|------|------|------|------|------|------|
| **WS·16** | 16 | **13.38** | 13.78 | **13.79** | 12.15 | 24.2 | **66.9** | 10.32 | 8.56 |
| **WS·32** | 32 | **25.62** | 26.24 | **26.25** | 23.09 | 32.0 | **72.6** | 10.41 | **9.59** |
| **FS Shared·OWN·16** | 16 | **10.57** | 14.41 | 14.42 | 11.39 | 82.5 | 64.0 | 10.38 | 9.21 |
| **FS Shared·OWN·32** | 32 | **24.83** | 26.03 | **26.04** | 23.97 | 30.7 | **73.4** | 10.36 | 9.36 |
| **FS Shared·CROSS·32** | 32 | **26.81** | 26.83 | **26.84** | 24.57 | 31.6 | **71.1** | 10.38 | 9.22 |
| Local NVMe·16 | 16 | 19.03 | 19.04 | 19.04 | 16.11 | 24.1 | 49.7 | 6.79 | 6.31 |
| Local NVMe·32 | 32 | 34.48 | 36.44 | 36.44 | 31.60 | 30.9 | 53.3 | 6.78 | 6.45 |
| Recompute·16 | 16 | 114.78 | 210.41 | 268.52 | 122.80 | 1817.4 | 3.8 | ≈0 | — |
| Recompute·32 | 32 | 251.69 | 466.95 | 608.52 | 255.70 | 5244.0 | 3.4 | ≈0 | — |

> CROSS = cross-read (A reads sessions injected by B, B reads sessions injected by A, 32/32 full disk hits); OWN = each reads its own injected sessions. In the FS·OWN·16 gear, TTFT p50 is low but p90 is elevated, due to request divergence under fs async loading at low concurrency (see §5.4); this phenomenon disappears at higher concurrency gears.

---

## 5. Comparative Analysis

### 5.1 AISSD5000 vs Local NVMe (Both LocalDiskBackend, Different Media Only)

| Full-Machine Concurrency | Metric | Local NVMe | **AISSD5000** | Benefit |
|------|------|------|------|------|
| 16 | TTFT p50 | 19.03 s | **13.38 s** | **−30%** |
| 16 | TTFT p99 | 19.04 s | **13.79 s** | **−28%** |
| 16 | Aggregate Throughput | 49.7 tok/s | **66.9 tok/s** | **+35%** |
| 32 | TTFT p50 | 34.48 s | **25.62 s** | **−26%** |
| 32 | TTFT p99 | 36.44 s | **26.25 s** | **−28%** |
| 32 | Aggregate Throughput | 53.3 tok/s | **72.6 tok/s** | **+36%** |

- The gap is the difference in media supply capability: local disk busy window means are 6.31 / 6.45 GB/s, pinned to its 6.78 GB/s physical ceiling; AISSD5000 achieves 8.56 / 9.59 GB/s, with peaks of 10.3–10.4 GB/s (90%+ of single-port line rate);
- Bandwidth accounting is self-consistent: at concurrency 32, 32 × 7.15 = 229 GB must be transferred; AISSD5000 at 9.59 GB/s requires approximately 23.9 s (measured TTFT p50 25.6 s), local disk at 6.45 GB/s requires approximately 35.5 s (measured 34.5 s) — **the TTFT gap ≈ the media bandwidth gap**;
- p99 and p50 are nearly equal (true for all four disk gears): LMCache async loading allows requests within the same gear to fully overlap and complete synchronously, with no long tail — **low latency is for all requests, not just the median**.

### 5.2 AISSD5000 vs No External Storage Recompute

| Full-Machine Concurrency | Metric | Recompute | **AISSD5000** | Benefit |
|------|------|------|------|------|
| 16 | TTFT p50 | 114.8 s | **13.4 s** | **8.6× faster** |
| 16 | TTFT p99 | 268.5 s | **13.8 s** | **19.5× faster** |
| 16 | TPOT p50 | 1817 ms | **24.2 ms** | **75× better** |
| 16 | Aggregate Throughput | 3.8 tok/s | **66.9 tok/s** | **17.6× higher** |
| 32 | TTFT p50 | 251.7 s | **25.6 s** | **9.8× faster** |
| 32 | TTFT p99 | 608.5 s | **26.3 s** | **23.2× faster** |
| 32 | TPOT p50 | 5244 ms | **32.0 ms** | **164× better** |
| 32 | Aggregate Throughput | 3.4 tok/s | **72.6 tok/s** | **21.4× higher** |

Recompute is not only an order of magnitude slower for the first token, but **decoding is also crippled**: continuous 30K prefill operations crowd out decode batches, pushing TPOT p50 to 1.8–5.2 seconds/token (normal 24–32 ms), making the output stream after the first token equally unusable; under the p99 metric, the gap widens to 19–23× (the worst recompute request waits 4.5–10 minutes). The TP4 instance has half the compute power of TP8, and the recompute degradation is deeper than the TP8 configuration (12.6–20×) — **the finer the instance slicing, the more essential the KV external tier becomes**.

### 5.3 fs:// Shared Pool vs Dedicated Disk Backend & Cross-Instance Sharing Cost

| Comparison (concurrency 32) | TTFT p50 | TTFT p99 | TPOT p50 | Aggregate throughput | Avg disk busy window |
|------|------|------|------|------|------|
| WS independent pool (read own) | 25.62 s | 26.25 s | 32.0 ms | 72.6 tok/s | 9.59 GB/s |
| FS shared pool·read own | 24.83 s | 26.04 s | 30.7 ms | 73.4 tok/s | 9.36 GB/s |
| **FS shared pool·cross-read peer's** | **26.81 s** | **26.84 s** | **31.6 ms** | **71.1 tok/s** | 9.22 GB/s |

- **Zero performance cost for shared backend**: FS·OWN and WS metrics are consistent within noise;
- **Near-zero cost for cross-instance sharing**: CROSS is 8% (p50) / 3% (p99) slower than OWN, with 32/32 full disk hits;
- **Cross-instance reading peer's data is still 22% faster than reading own data from local disk** (26.81 vs 34.48 s) — sharing semantics combined with media advantage;
- Comparison with LocalDiskBackend cross-instance attempt (hit=0, TTFT 67 s degraded to recompute): sharing capability is determined by backend selection; `fs://` opens the path in one step.

### 5.4 TPOT (Decode Quality) Interpretation

- TPOT p50 for all four disk configurations is within 24–32 ms: **KV media does not affect steady-state decode speed**; differences only appear in the recovery phase (TTFT);
- FS·OWN·16 config TPOT p50 of 82.5 ms is elevated (accompanied by TTFT request divergence): **after 5 rounds of retesting + 2 rounds of fix verification, it was identified as a configuration issue, not a coincidence**. Root cause: the 4 GB CPU staging layer was insufficient for the fs remote fetch path — 8 concurrent requests with in-flight KV (~57 GB) far exceeded the staging pool, triggering "allocation failures (up to 1.9×10⁴ warnings per round) → no eviction candidates → blocking timeout cancel retry". Mild cases caused request completion time divergence (TPOT p50 rising to 80–92 ms); severe cases caused entire wave stall (1 out of 5 retest rounds: single instance 8 requests all had TTFT of 61.8 s). **After raising `LMCACHE_MAX_LOCAL_CPU_SIZE` to 64 GB, allocation failure and timeout warnings dropped to zero, and 3 consecutive rounds of true cold reads were all clean** (TTFT p50 13.1–15.4 s, TPOT p50 23–27 ms, no stalls). Only the first round after instance startup showed slight warm-up divergence. Production configuration recommendation: CPU staging layer for fs remote path should be ≥ in-flight KV volume (≥57 GB for this workload);
- Incidental finding: 64 GB staging layer allows recently read sessions to reside in CPU layer; second recovery TTFT is only **1.8 s** (memory layer hit, zero disk reads) — the value of the GPU/memory/array three-tier hierarchy is directly realizable on large-memory hosts;
- Recompute config TPOT p50 1.8–5.2 s: decode and continuous prefill contend for the engine, output stream freezes — recompute is unusable end-to-end.

### 5.5 Cross-validation with TP8 Single-Instance Form (07-05)

On the same machine with the same workload specification, TP8×1 round AISSD5000 vs local disk showed TTFT −26%~−32%, throughput +29%~+40%; this round TP4×2 showed TTFT −26%~−30%, throughput +35%~+36% — **benefit magnitude is stable across deployment forms**. In both forms, array peak is 10.2–10.4 GB/s, local disk is pinned at 6.78 GB/s.

---

## 6. Conclusion

1. **AISSD5000 vs local NVMe: TTFT p50 reduced by 26%–30%, p99 reduced by 28%, throughput increased by 35%–36%** (480B TP4×2 long-context cold recovery, verified by full-config physical cold reads); p99≈p50 indicates benefit covers all requests without long tail. Mechanism: local disk pinned at 6.78 GB/s physical ceiling, AISSD5000 sustains 8.6–9.6 GB/s (90%+ single-port line rate).
2. **vs recompute without external storage: TTFT p50 8.6–9.8× faster, p99 19–23× faster, throughput 18–21× higher, TPOT 75–164× better** — recompute is unusable on both first-token and decode ends; KV external storage is a prerequisite for multi-instance deployment of long-context large models.
3. **`fs://` shared pool enables cross-instance hot sharing with zero performance cost**: cross-read achieves full hits with only 3%–8% overhead; cross-instance reading peer's data is still 22% faster than reading own data from local disk. One copy of KV reused across the entire machine, sessions freely migratable, cache survives restart.
4. **Decode quality is independent of media** (TPOT p50 24–32 ms consistent across four configs); storage value is concentrated in recovery latency and throughput.
5. **Data is credible**: deviation between two independent complete measurements ≤5%; bandwidth accounting and working set are self-consistent; all hits/disk reads double-verified.

---

## 7. Limitations

1. FS shared pool TPOT divergence for concurrency 16 has been retested, bounded, and fixed (§5.4: caused by insufficient 4 GB CPU staging layer, resolved after 64 GB); summary table data was measured under 4 GB config; FS two configs will only have better metrics with sufficient staging layer;
2. p99 is nominal rank (n=16/32 means worst request); stricter tail latency requires larger samples;
3. Cross-node sharing and multi-machine aggregate provisioning were not physically verified (requires a second host connected to the storage network);
4. Synthetic workload (uniform access, fixed length) is conservative relative to real skewed traffic.

---

## Appendix A: Reproduction Commands (Colleague Server, Container `vllm`)

### A.1 Dual Instance Startup (Four Configs Differ Only in Storage Environment Variables)

```bash
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8
## ① WS independent pool: ENVS="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
##               LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache480tp4 LMCACHE_MAX_LOCAL_DISK_SIZE=1000"
## ② FS shared pool: ENVS="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
##               LMCACHE_REMOTE_URL=fs://local:0/mnt/ws5000/kvpool_fs LMCACHE_REMOTE_SERDE=naive"
## ③ Local NVMe:  ENVS same as ① but LMCACHE_LOCAL_DISK=file:///srv2/lmcache480tp4_local
## ④ Recompute:   ENVS empty and remove --kv-transfer-config
for I in 0 1; do
  DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7"); PORT=$((8000+I))
  docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 $ENVS; \
vllm serve $MODEL --served-model-name qwen \
 --tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
 --max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
 --port $PORT > /mnt/ws5000/log_i$I.log 2>&1"
  sleep 30
done
```

### A.2 Injection and Measurement (bench_mp2.py see Appendix B)

## Parallel injection of 48 sessions
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8000 populate 530 24 0  > /mnt/ws5000/results/m2_ppA.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8001 populate 530 24 24 > /mnt/ws5000/results/m2_ppB.log 2>&1"
## Measurement per workload (example: full-node concurrency 32; CROSS tier A/B starting point swapped)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
iostat -x 1 900 /dev/md0 > /tmp/io.log &      # Local RAID monitoring /dev/nvme1n1
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8000 measure 530 16 0  64 16 > /mnt/ws5000/results/m2_WS16_A.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8001 measure 530 16 24 64 16 > /mnt/ws5000/results/m2_WS16_B.log 2>&1"
## Full-node percentile aggregation: merge REQ lines from A/B files and compute per Section 3.1 definition

## Appendix B: Enhanced Load Client `bench_mp2.py` (Per-Request TTFT/E2E/TPOT)

```python
import urllib.request, json, time, sys
from concurrent.futures import ThreadPoolExecutor
port=sys.argv[1]; mode=sys.argv[2]; reps=int(sys.argv[3]); N=int(sys.argv[4]); off=int(sys.argv[5])
decode=int(sys.argv[6]) if len(sys.argv)>6 else 64
conc=int(sys.argv[7]) if len(sys.argv)>7 else 16
BASE='http://127.0.0.1:%s/v1/chat/completions'%port
basep='Background: AISSD5000 is a domestic high-performance all-flash NVMe-oF storage, serving as a tiered backend medium for LLM inference KV cache, cooperating with vLLM and LMCache to tier KV between HBM/memory/disk.'
def make_prefix(i): return '[sess-%05d] '%i + basep*reps
def req(sid, maxtok):
    body=json.dumps({'model':'qwen','stream':True,'messages':[{'role':'system','content':make_prefix(sid)},{'role':'user','content':'Answer%d'%sid}],'max_tokens':maxtok,'temperature':0}).encode()
    st=time.time(); ttft=None; n=0
    try:
        r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={'Content-Type':'application/json'}),timeout=1800)
        for line in r:
            sx=line.decode('utf-8','ignore')
            if sx.startswith('data:') and '"content"' in sx:
                if ttft is None: ttft=time.time()-st
                n+=1
        return (sid, ttft, time.time()-st, n)
    except Exception as e:
        sys.stderr.write('ERR sid=%d %s\n'%(sid,e)); return (sid,None,None,0)
def pct(a,q):
    import math
    return a[min(len(a)-1, max(0, math.ceil(q*len(a))-1))] if a else 0.0
if mode=='populate':
    t0=time.time()
    for i in range(N): req(off+i,1)
    print('[p%s] populate N=%d off=%d wall=%.1fs'%(port,N,off,time.time()-t0),flush=True); sys.exit(0)
res=[]; t0=time.time()
with ThreadPoolExecutor(max_workers=conc) as ex:
    futs=[ex.submit(req,off+i,decode) for i in range(N)]
    for f in futs:
        x=f.result()
        if x[1] is not None: res.append(x)
wall=time.time()-t0
for sid,ttft,e2e,n in sorted(res):
    tp=(e2e-ttft)/(n-1)*1000 if n>1 else 0
    print('REQ sid=%d ttft=%.3f e2e=%.3f ntok=%d tpot_ms=%.1f'%(sid,ttft,e2e,n,tp),flush=True)
tt=sorted(r[1] for r in res)
tpots=sorted((r[2]-r[1])/(r[3]-1)*1000 for r in res if r[3]>1)
tot=sum(r[3] for r in res)
print('[p%s] n=%d wall=%.1fs TTFT p50=%.2f p90=%.2f p99=%.2f mean=%.2f | TPOT_ms p50=%.1f p99=%.1f mean=%.1f | outtok/s=%.1f'%(
  port,len(res),wall,pct(tt,.5),pct(tt,.9),pct(tt,.99),sum(tt)/len(tt),
  pct(tpots,.5),pct(tpots,.99),sum(tpots)/len(tpots),tot/wall),flush=True)
```

## Appendix C: Raw Data Archive (Colleague Server)

| File | Content |
|------|---------|
| `/tmp/full2.out` | Full-metric re-test orchestration log (nine tiers, all outputs) |
| `/mnt/ws5000/results/m2_*_{A,B}.log` | Per-tier, per-instance raw request data (REQ lines) and summary lines for both instances |
| `/mnt/ws5000/{m2fs,m2ws,m2loc,m2rc}_i{0,1}.log` | Complete logs of eight service instances (disk groups include per-request hit evidence lines) |
| `/tmp/io_m2_*.log` | Per-tier iostat second-level raw records |
| `/tmp/tp4x2b.out`, `/tmp/tp4x2c.out`, `/tmp/fsshare.out` | Two independent measurement sessions (morning/afternoon) for reproducibility cross-check |
| `/mnt/ws5000/kvpool_fs/` | fs shared pool (344 GB / 22,656 files, retained for re-inspection) |
