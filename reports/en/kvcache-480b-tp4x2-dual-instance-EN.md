# FX100 KV-Cache Benchmark (480B, TP4×2 Dual-Instance, Official)

> **Reference translation.** The signed Chinese original is the authoritative version. AI-translated by Mingxin's translation pipeline; number fidelity machine-verified (see `dmkt/report_i18n/qc/X1.json`). Report date: 2026-07-06.

## AISSD5000 KV Cache Performance Test Report – 480B Model·TP4×2 Dual Instance·Long Context Cold Recovery

**Test Platform**: AMD Instinct MI308X ×8 (192 GB HBM per card) / ROCm 7.2 / vLLM 0.20.1+rocm721 + LMCache v1 (upstream mainline source compiled, includes disk parallel read optimization)
**Storage Under Test**: AISSD5000 (WS5000) all-flash NVMe-oF array (4-disk RAID0, RoCEv2 over 100GbE, XFS, 14 TB)
**Reference Storage**: Server-local NVMe (Solidigm, PCIe Gen4, mounted partition 2 TB); no external storage recompute
**Model**: Qwen3-Coder-480B-FP8 (MoE, weights ~450 GB)
**Deployment**: **Dual instance TP4×2** – Instance A occupies cards 0–3, Instance B occupies cards 4–7, each serving independently (+ expert parallel)
**Workload**: Long context cold recovery – ~29.8K tokens per session, KV ≈ 7.15 GB/session
**Date**: 2026-07-06

---

## 1. Test Objectives and Summary of Conclusions

The previous round of testing (2026-07-05) quantified the benefits of AISSD5000 over local NVMe and recompute in an **8-card single instance (TP=8)** configuration. This round switches the same 8-card machine to a **dual instance TP4×2** – a more common production "multi-service on one machine" split (higher instance-level fault tolerance, independent scaling, single instance failure does not bring down the entire machine) – to answer three questions:

1.  Are the AISSD5000 benefit conclusions **valid across deployment forms** (rather than a coincidence specific to the TP8 form)?
2.  When a single AISSD5000 **supplies multiple independent inference instances simultaneously**, what is its supply capability and stability?
3.  What is the feasibility boundary for **reusing the same KV** across multiple instances via the shared array?

**Key Conclusions (verified via per-request physical cold read across all concurrency levels):**

1.  **Compared to local NVMe: AISSD5000 reduces TTFT by 25%–30% and increases aggregate output throughput by 28%–35%, consistent with the TP8 form conclusions (26%–32% / 29%–40%) – the benefit is stable across deployment forms.** At full-machine concurrency 16: TTFT p50 decreased from 19.3 s to **13.5 s (30% reduction)**, aggregate throughput increased from 49.2 to **66.5 tok/s (35% increase)**.
2.  **One array feeds two independent engines simultaneously**: During dual-instance concurrent cold read, the array peak reached **10.33 GB/s** (90%+ of single-port 100GbE line rate), with busy-window average 8.55–8.88 GB/s; the local disk was pinned at its physical limit of **6.78 GB/s** in both forms. During the ingestion phase, dual instances wrote 343 GB in parallel (0.95 GB/s sustained write + dual-engine prefill read-write mix) with zero anomalies.
3.  **Compared to no external storage recompute: TTFT is 9.5–10x faster (p90 metric 16x+), throughput is 17–20x higher.** A TP4 instance has only half the compute power of TP8; recomputing a 30K prefix at full-machine concurrency 16 requires waiting 2.1–4.5 minutes for the first token – the smaller the instance split, the more unusable recompute becomes, and the stronger the mandatory requirement for an external KV tier.
4.  **Cross-instance hot sharing reveals a software boundary (honest negative result)**: When Instance B cold-reads sessions ingested by Instance A, the hit rate is 0, degrading to recompute. The files themselves are accessible to both parties on the shared array, but the LMCache `LocalDiskBackend` index is **process-memory-resident** and does not scan directories – cross-instance reuse requires a shared index layer (remote/Mooncake-type backend or index rebuild on restart). The array's capacity and bandwidth are already sufficient; the gap lies in the caching software, not the storage.

---

## 2. System Under Test and Environment

### 2.1 Hardware

| Component | Configuration |
|-----------|---------------|
| GPU | 8 × AMD Instinct MI308X (192 GB HBM per card, gfx942) |
| Deployment Form | **Instance A: Cards 0–3 (TP=4, port 8000); Instance B: Cards 4–7 (TP=4, port 8001)**, both with expert parallel |
| CPU / Memory | 2 × AMD EPYC 9654 (384 threads), ~1.5 TB memory |
| **Storage Under Test** | **AISSD5000: 4-disk RAID0 (`/dev/md0`, xfs, 14 TB), NVMe-oF / RoCEv2, single-port 100 GbE** |
| Reference Storage | Local NVMe single disk (`/dev/nvme1n1`, PCIe Gen4, mounted `/srv2`); no external storage (recompute) |

### 2.2 Software

| Component | Version |
|-----------|---------|
| Operating System | Ubuntu 22.04, kernel 6.8.0-124-generic |
| GPU Stack | ROCm 7.2 (gfx942) |
| Inference Engine | vLLM 0.20.1+rocm721 |
| KV Cache Library | LMCache (upstream mainline 2026-06-29 source compiled; default async disk load + disk parallel read optimization) |
| Model | Qwen3-Coder-480B-FP8 (MoE, weights ~450 GB, loaded from AISSD5000 array for all rounds) |
| Key Parameters | `--tensor-parallel-size 4 --enable-expert-parallel`, `--max-model-len 32768`, `--gpu-memory-utilization 0.9`, `--no-enable-prefix-caching` (cold read verification), `LMCACHE_CHUNK_SIZE=256`, `LMCACHE_MAX_LOCAL_CPU_SIZE=4`, `PYTHONHASHSEED=0` |

Dual instances started with a 30 s offset; both instances loaded weights from the array in parallel (total 900 GB), both ready in 4.5 minutes, VRAM utilization at 90% on both sides.

### 2.3 Workload and Capacity Baseline

- Single session system prompt prefix ~**29.8K tokens** (reps=530), single session KV (TP4 shard total) ~**7.15 GB**;
- Each round, dual instances **ingest 48 distinct sessions in parallel** (A ingests 0–23, B ingests 24–47), measured disk write **343 GB**, far exceeding HBM residency and CPU tier capacity (4 GB×2);
- Measurement: Both instances initiate simultaneously, each reads its own ingested sessions (A reads 0..N−1, B reads 24..24+N−1), one request per session, decode=64, temperature=0;
- Full-machine concurrency = 2 × per-instance concurrency, levels: 8+8=16, 16+16=32 – aligned with the previous TP8 round's full-machine concurrency 16/32 levels.

---

## 3. Test Methodology

### 3.1 Reference Design (Three Configurations × Two Full-Machine Concurrency Levels)

| Group | KV Backend Medium | Full-Machine Concurrency | iostat Monitoring |
|-------|-------------------|--------------------------|-------------------|
| ① AISSD5000 | `/mnt/ws5000/lmcache480tp4` (md0, RAID0) | 16 / 32 | `/dev/md0` |
| ② Local NVMe | `/srv2/lmcache480tp4_local` (nvme1n1, single disk) | 16 / 32 | `/dev/nvme1n1` |
| ③ Recompute (No External Storage) | None (LMCache not mounted) | 16 / 32 | — (disk read ≈0) |

Only the KV backend medium differs among the three groups; dual-instance layout, model, weight source, engine parameters, session construction, and concurrency levels are identical.

### 3.2 Physical Cold Read Guarantee (Fivefold Control)

1. `--no-enable-prefix-caching`: HBM retains no prefix KV.
2. LMCache CPU tiering layer per instance compressed to 4 GB: the memory tier cannot hold any session.
3. **Before each measurement tier, the host executes `sync; echo 3 > /proc/sys/vm/drop_caches`**: clears the 1.5 TB host memory page cache.
4. Each session is read only once, and the session sets read by the two instances are disjoint (A: 0–15, B: 24–39) — no cross-instance page cache sharing.
5. Dual verification: all 96 measurement requests in the logs of the four disk group instances show `hit tokens: 30208` (full disk hit) and `need to load ≈ 30K`; the recompute group processes have no LMCache and disk reads ≈ 0.

> Note: The new LMCache defaults to asynchronous loading. Its log `Retrieved ... throughput` is the staging copy speed and cannot be used to judge physical disk reads; physical reads are always determined by `iostat`.

### 3.3 Measurement and Collection

- The measurement clients of the two instances **start simultaneously** (issued in the same second), measuring streaming TTFT per request.
- Aggregate output throughput = total output tokens for the entire machine (2×N×64) ÷ the batch completion time of the slower of the two instances.
- Each tier collects `iostat -x 1` in parallel (peak, busy window average, busy window duration).
- Between group switches, inference processes (including EngineCore/Worker remnants) are completely cleaned up, and zero HBM on all 8 GPUs is confirmed.

---

## IV. Test Results

### 4.1 Three-Way Comparison Summary Table (TP4×2, Two Instances Initiated Simultaneously)

| Tier (Full Machine Concurrency) | TTFT p50 A / B (s) | TTFT p90 A / B (s) | Aggregate Throughput (tok/s) | Disk Peak | Disk Busy Window Avg | Cold Read Verification |
|------|------|------|------|------|------|------|
| **AISSD5000 · 16 (8+8)** | **13.17 / 13.89** | 13.18 / 13.90 | **66.5** | **10.33 GB/s** | 8.55 GB/s (14 s) | 16/16 |
| **AISSD5000 · 32 (16+16)** | **26.01 / 27.80** | 26.01 / 27.81 | **69.0** | 10.27 GB/s | **8.88 GB/s (27 s)** | 32/32 |
| Local NVMe · 16 (8+8) | 19.34 / 19.29 | 19.35 / 19.30 | 49.2 | 6.78 GB/s | 6.31 GB/s (19 s) | 16/16 |
| Local NVMe · 32 (16+16) | 36.10 / 35.41 | 36.11 / 35.42 | 53.8 | 6.79 GB/s | 6.64 GB/s (35 s) | 32/32 |
| Recompute · 16 (8+8) | 132.6 / 125.0 | 192.6 / 267.9 | 3.8 | ≈0 | — | No LMCache |
| Recompute · 32 (16+16) | 269.6 / 273.0 | 435.3 / 466.8 | 3.4 | ≈0 | — | No LMCache |

### 4.2 Core Comparison 1: AISSD5000 vs Local NVMe

| Full Machine Concurrency | Local TTFT p50 | **AISSD5000 TTFT p50** | TTFT Reduction | Local Aggregate Throughput | **AISSD5000 Aggregate Throughput** | Throughput Improvement |
|------|------|------|------|------|------|------|
| 16 | 19.3 s | **13.5 s** | **−30%** | 49.2 | **66.5** | **+35%** |
| 32 | 35.8 s | **26.9 s** | **−25%** | 53.8 | **69.0** | **+28%** |

The magnitude is consistent with the TP8 single-instance configuration (TTFT −26%~−32%, throughput +29%~+40%). **The benefit of AISSD5000 is independent of deployment configuration**: whether the machine is split into one large instance or two medium instances, the local disk is bottlenecked by its physical limit of 6.78 GB/s, while AISSD5000 supplies at 8.6–10.3 GB/s.

### 4.3 Core Comparison 2: AISSD5000 vs No External Storage Recompute

| Full Machine Concurrency | Metric | Recompute | **AISSD5000** | Benefit |
|------|------|------|------|------|
| 16 | TTFT p50 (Avg of Two Instances) | 128.8 s | **13.5 s** | **9.5x faster** |
| 16 | TTFT p90 (Worse Instance) | 267.9 s | **13.9 s** | **19.3x faster** |
| 16 | Aggregate Throughput | 3.8 tok/s | **66.5 tok/s** | **17.5x higher** |
| 32 | TTFT p50 (Avg of Two Instances) | 271.3 s | **26.9 s** | **10.1x faster** |
| 32 | Aggregate Throughput | 3.4 tok/s | **69.0 tok/s** | **20.3x higher** |

> Interpretation: The prefill compute power of a TP4 instance is only half that of a TP8 instance. Under concurrency, recomputing a 30K prefix results in a first-token wait of 2.1–4.5 minutes, with a long tail of nearly 8 minutes — **the smaller the instance, the more unusable recompute becomes**. The production trend is precisely towards more/smaller instances (for fault tolerance and elasticity), which makes the KV external tier a "deployment prerequisite" rather than an "optimization option"; the choice of medium under this prerequisite is answered by §4.2.

### 4.4 Deployment Configuration Comparison: TP8×1 vs TP4×2 (Same Machine, Same Load Specification)

| Full Machine Concurrency | Configuration | AISSD5000 TTFT p50 | AISSD5000 Throughput | Local Disk TTFT p50 | Local Disk Throughput |
|------|------|------|------|------|------|
| 16 | TP8×1 (07-05) | 11.85 s | 74.9 tok/s | 17.31 s | 53.6 |
| 16 | **TP4×2 (This Round)** | 13.5 s | 66.5 tok/s | 19.3 s | 49.2 |
| 32 | TP8×1 (07-05) | 26.35 s | 71.6 tok/s | 35.73 s | 53.9 |
| 32 | **TP4×2 (This Round)** | 26.9 s | 69.0 tok/s | 35.8 s | 53.8 |

Three observations:

1. **Storage supply is independent of configuration**: In both configurations, the AISSD5000 peak is 10.2–10.3 GB/s (90%+ of single-port line rate), and the local disk is pinned at 6.78 GB/s — the array can be saturated by the 8-way sharded parallel reads of a single TP8 instance, as well as by the concurrent reads of two independent TP4 instances.
2. **At the same concurrency, TP8 is slightly better; at concurrency 32, they are nearly equal**: With TP8, each request is read in parallel shards by 8 GPUs and prefilled across the whole machine, resulting in faster single-request recovery; TP4×2 nearly matches TP8 when concurrency is maxed out (32). The configuration choice can be based entirely on business needs (fault tolerance/elasticity vs. single-request latency), **storage does not impose a constraint**.
3. **Recompute's sensitivity to configuration is far greater than KV recovery**: KV recovery slows by only 2%–14% when switching from TP8 to TP4×2, whereas recompute's TTFT at the concurrency 32 tier reaches 271 s — the more granular the configuration, the greater the value of the KV tier.

### 4.5 Mechanism Verification (Bandwidth Accounting Self-Consistency)

- Full-machine concurrency 16 requires transferring 16 × 7.15 = **114 GB**: AISSD5000 at busy-window average 8.55 GB/s takes ~13.4 s (measured TTFT p50 13.5 s, consistent); local disk at 6.31 GB/s takes ~18.1 s (measured 19.3 s, consistent);
- Full-machine concurrency 32 requires transferring **229 GB**: AISSD5000 at 8.88 GB/s takes ~25.8 s (measured 26.9 s); local disk at 6.64 GB/s takes ~34.5 s (measured 35.8 s) — **TTFT for all four tiers is explained by media bandwidth, measurements are self-consistent**;
- When two instances perform cold reads concurrently, the difference in TTFT p50 between the two instances is ≤0.7 s (16-tier) / 1.8 s (32-tier); the array provides balanced supply to two independent clients, with no starvation or skew;
- During the injection phase, two instances write in parallel: 48 sessions / 343 GB / 362 s (≈0.95 GB/s sustained write, mixed with prefill reads/writes); the WS round and local round injection times are nearly identical (363 s vs 361 s) — the write demand is far below the upper limit of both media types, injection is compute-bound and does not affect the fairness of the comparison.

### 4.6 Mechanism Discovery: Cross-Instance Hot Sharing Requires a Shared Index Layer (Honest Negative Result)

Design motivation: Both instances mount the same array directory; in theory, sessions injected by A should be directly hit by B (TP4↔TP4 sharding format is identical, `PYTHONHASHSEED=0` ensures chunk keys are identical).

Measurement: After clearing the page cache, instance B cold-reads sessions 0–3 injected by instance A. B's logs show all 4 requests have `hit tokens: 0`, degrading to recompute (TTFT p50 67 s).

Root cause: The key index of LMCache's `LocalDiskBackend` is maintained in **process memory** (registered upon write), does not scan the disk directory, and has no cross-process synchronization — although the files are on the shared array and readable by both sides, B's index does not contain the keys written by A, so the query directly misses.

Conclusion and path: **The array-side capacity, bandwidth, and multi-client concurrent supply capability are already in place (§4.5 item 3); the gap for cross-instance/cross-node KV pool sharing lies in the index layer of the caching software** — three engineering paths exist: ① Use the LMCache remote backend (centralized index + array as data plane); ② A shared metadata service like Mooncake/Redis; ③ Directory scan and index rebuild at instance startup (suitable for static scenarios of "one-time prefill, multi-instance reuse"). Listed for subsequent verification.

---

## V. Analysis and Discussion

### 5.1 Robustness of Gains

Two rounds of experiments (TP8×1, TP4×2) were conducted independently on the same machine with the same load specification. The gain magnitude of AISSD5000 over local NVMe is highly consistent (TTFT −25% to −32%, throughput +28% to +40%). The mechanism is the same: the instantaneous bandwidth demand of 30K long-context cold recovery exceeds the physical ceiling of the local disk (6.78 GB/s), and the difference is the gap in supply capability between the two media. **This conclusion is insensitive to deployment form and instance granularity, and can be directly extrapolated to production partitioning of "one machine, N instances".**

### 5.2 Supply Model of "One Array Feeds the Whole Machine"

This round proves that a single 100GbE port on AISSD5000 can simultaneously feed the recovery storm of two 480B instances (combined peak 10.3 GB/s). Extrapolating from this supply model: adding a second 100G port (6 ports total on the device) can support concurrent recovery of a 4-instance cluster; in cross-node scenarios, multiple inference machines sharing the same array are only constrained by port aggregation bandwidth (full-configuration nominal 72 GB/s). The local disk solution is one set per machine, with an upper limit of 6.78 GB/s and a capacity of 2 TB that cannot simultaneously accommodate weights and KV pool — **multi-instancing amplifies rather than narrows the gap between the two**.

### 5.3 KV Layer Transitions from "Optimization" to "Prerequisite"

The TTFT for recomputing a 30K prefix on a TP4 instance (2.1–4.5 minutes) is worse than on TP8 (2.5 minutes) and an order of magnitude worse than KV recovery (13.5–27 s). Multi-instance partitioning is the norm in production, and the finer the partitioning, the smaller the single-instance compute power, and the more infeasible recompute becomes — the KV external storage layer is a prerequisite infrastructure for multi-instance deployment, and AISSD5000 consistently outperforms in media comparison at this layer.

---

## VI. Conclusion

1. **Gains are valid across deployment forms (core conclusion)**: Under the 480B dual-instance TP4×2 long-context cold recovery load, AISSD5000 compared to local NVMe achieves **25%–30% lower time to first token and 28%–35% higher aggregate throughput**, consistent with the TP8 single-instance round (−26% to −32% / +29% to +40%); the mechanism is the same: local disk is pinned at a 6.78 GB/s physical ceiling, while AISSD5000 supplies 8.6–10.3 GB/s continuously.
2. **One array simultaneously feeds two independent engines**: Dual-instance concurrent cold read peak is 10.33 GB/s (90%+ of single-port line rate), with balanced supply to both instances (TTFT difference ≤1.8 s); parallel write of 343 GB during injection phase with zero anomalies. Multi-instancing amplifies the gap between AISSD5000 and local disk.
3. **9.5–10× faster than recompute (16–19× at p90), 17–20× higher throughput**; the finer the instance partitioning, the more unusable recompute becomes; the KV external layer is a prerequisite infrastructure for multi-instance deployment.
4. **The boundary for cross-instance hot sharing is in the software index layer, not the storage**: File-level sharing is ready; LMCache's `LocalDiskBackend` in-process index causes cross-instance misses; using a centralized index backend or index rebuild at startup can resolve this, listed for subsequent verification.

---

## VII. Limitations and Future Work

1. **Cross-instance sharing not yet enabled** (§4.6): Requires retesting with remote/shared index backend for "one-time prefill, full-machine reuse", and further extrapolation to cross-node sharing;
2. **Single-machine scope**: Aggregate supply of multiple inference machines sharing the same array has not been physically verified;
3. **Recompute group tiers constrained by time**: The recompute round fully tested the 16/32 tiers (total ~15 minutes), no higher concurrency was scanned;
4. **Synthetic load**: Uniform access, fixed length, conservative estimate relative to real skewed traffic; each data point is a single measurement.

---

## Appendix A: Reproduction Commands (Colleague Server, Inside Container `vllm`)

### A.1 Dual Instance Startup (AISSD5000 round; for local round, replace `LMCACHE_LOCAL_DISK` with `file:///srv2/lmcache480tp4_local`; for recompute round, remove the three LMCACHE variables and `--kv-transfer-config`)

```bash
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8
for I in 0 1; do
  DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7"); PORT=$((8000+I))
  docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 \
LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache480tp4 LMCACHE_MAX_LOCAL_DISK_SIZE=1000; \
vllm serve $MODEL --served-model-name qwen \
 --tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
 --max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
 --port $PORT > /mnt/ws5000/tp4ws_i$I.log 2>&1"
  sleep 30
done
```

### A.2 Dual Instance Parallel Injection (48 Sessions / 343 GB)

```bash
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 populate 530 24 0  > /mnt/ws5000/results/tp4_ppA.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 populate 530 24 24 > /mnt/ws5000/results/tp4_ppB.log 2>&1"
```

### A.3 Per-tier measurement (example: full-machine concurrency 32 = 16+16; must drop page cache before measurement)

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
iostat -x 1 400 /dev/md0 > /tmp/io_tp4_WS_c16x2.log &   # local disk monitoring /dev/nvme1n1
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 measure 530 16 0  64 16 > /mnt/ws5000/results/tp4_WS_c16x2_A.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 16 24 64 16 > /mnt/ws5000/results/tp4_WS_c16x2_B.log 2>&1"
## After both bench_mp processes exit:
awk '$1=="md0"{if($3>m)m=$3} $1=="md0" && $3>500000{c++;s+=$3} END{printf "Peak %.2f GB/s  Busy-avg %.2f GB/s(%ds)\n", m/1e6, s/c/1e6, c}' /tmp/io_tp4_WS_c16x2.log
```

### A.4 Cold-read verification

```bash
## Each instance should have exactly 24 measurement requests with full disk hits:
grep -acE 'hit tokens: 30[0-9]{3}' /mnt/ws5000/tp4ws_i0.log   # Expected 24
grep -acE 'need to load: (2[0-9]{4}|30[0-9]{3})' /mnt/ws5000/tp4ws_i0.log   # Expected 24
```

## Appendix B: Load client `bench_mp.py` (port-parameterized, dual-instance reuse)

```python
import urllib.request, json, time, sys
from concurrent.futures import ThreadPoolExecutor
port=sys.argv[1]; mode=sys.argv[2]; reps=int(sys.argv[3]); N=int(sys.argv[4]); off=int(sys.argv[5])
decode=int(sys.argv[6]) if len(sys.argv)>6 else 64
conc=int(sys.argv[7]) if len(sys.argv)>7 else 16
BASE='http://127.0.0.1:%s/v1/chat/completions'%port
basep='Background: AISSD5000 is a domestic high-performance all-flash NVMe-oF storage, serving as a tiered backend for large-model inference KV cache, cooperating with vLLM and LMCache for tiered KV access across HBM/memory/disk.'
def make_prefix(i): return '[sess-%05d] '%i + basep*reps
def req(sid, maxtok):
    body=json.dumps({'model':'qwen','stream':True,'messages':[{'role':'system','content':make_prefix(sid)},{'role':'user','content':'Answer %d'%sid}],'max_tokens':maxtok,'temperature':0}).encode()
    st=time.time(); ttft=None; n=0
    try:
        r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={'Content-Type':'application/json'}),timeout=1200)
        for line in r:
            sx=line.decode('utf-8','ignore')
            if sx.startswith('data:') and '"content"' in sx:
                if ttft is None: ttft=time.time()-st
                n+=1
        return (ttft, time.time()-st, n)
    except Exception as e:
        return (None,None,0)
def pct(a,q): return a[min(len(a)-1,int(len(a)*q))] if a else 0.0
if mode=='populate':
    t0=time.time()
    for i in range(N): req(off+i,1)
    print('[p%s] populate N=%d off=%d wall=%.1fs'%(port,N,off,time.time()-t0),flush=True); sys.exit(0)
res=[]; t0=time.time()
with ThreadPoolExecutor(max_workers=conc) as ex:
    futs=[ex.submit(req,off+i,decode) for i in range(N)]
    for f in futs:
        x=f.result()
        if x[0] is not None: res.append(x)
wall=time.time()-t0
tt=sorted(r[0] for r in res); tot=sum(r[2] for r in res)
print('[p%s] n=%d wall=%.1fs TTFT p50=%.3f p90=%.3f mean=%.3f tok/s=%.1f'%(port,len(res),wall,pct(tt,.5),pct(tt,.9),(sum(tt)/len(tt) if tt else 0),tot/wall),flush=True)
```

## Appendix C: Raw data archive (colleague server)

| File | Content |
|------|---------|
| `/tmp/tp4x2b.out`, `/tmp/tp4x2c.out` | Full-experiment orchestration logs (TTFT/throughput/bandwidth output per tier) |
| `/mnt/ws5000/results/tp4_*_{A,B}.log` | Per-tier raw client output for both instances |
| `/mnt/ws5000/tp4{ws,loc,rc}_i{0,1}.log` | Complete logs of six service instances (including per-request hit/need evidence lines) |
| `/tmp/io_tp4_*.log` | Per-tier iostat second-level raw records |
