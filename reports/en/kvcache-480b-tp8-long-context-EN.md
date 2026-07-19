# FX100 KV-Cache Benchmark (480B, TP8 Long-Context, Signed Official)

> **Reference translation.** The signed Chinese original is the authoritative version ([download](https://mingxinstorage.xyz/evidence/R2-kvcache-480b-tp8.pdf)). AI-translated by Mingxin's translation pipeline; number fidelity machine-verified (see `dmkt/report_i18n/qc/R2.json`). Report date: 2026-07-05.

# AISSD5000 KV Cache Performance Test Report – 480B Model · 8-GPU Single Instance · Long Context Cold Recovery

**Test Platform**: AMD Instinct MI308X ×8 (192 GB HBM per card) / ROCm 7.2 / vLLM 0.20.1+rocm721 + LMCache v1 (upstream mainline source compiled, includes disk parallel read optimization)
**Storage Under Test**: AISSD5000 (WS5000) all-flash NVMe-oF array (4-disk RAID0, RoCEv2 over 100GbE, XFS, 14 TB)
**Reference Storage**: Server local NVMe (Solidigm, PCIe Gen4, mounted partition 2 TB); No external storage recompute
**Model**: Qwen3-Coder-480B-FP8 (MoE, tensor parallel TP=8 + expert parallel, weights ~450 GB)
**Workload**: Long context cold recovery – ~29.8K tokens per session, KV ≈ 7.15 GB/session
**Date**: 2026-07-05

---

## 1. Test Objectives and Summary of Conclusions

Under the standard production deployment form of an 8-GPU single instance (TP=8) for a 480B-class large model, using **long context session cold recovery** as the workload (simulating Agent/code assistant long history session recovery storms: service restart, session migration, inactive session wake-up), quantify the performance benefits of AISSD5000 as a KV cache tiering backend medium relative to local NVMe and no external storage recompute, and provide the concurrency scaling curve.

**Main Conclusions (verified by per-request physical cold reads across all tiers):**

1. **Compared to local NVMe: AISSD5000 reduces time to first token (TTFT) by 26%–32% and increases output throughput by 29%–40%, stable across the 8–32 concurrency range**. At concurrency 16: TTFT p50 drops from 17.31 s to **11.85 s (32% reduction)**, output throughput increases from 53.6 to **74.9 tok/s (40% increase)**. The mechanism is pinned by `iostat` physical read measurements: under the same workload, the local disk's three tiers are all pinned at its physical limit of **6.78 GB/s**, while AISSD5000's three tiers peak at **10.2 GB/s** and sustain **9.29 GB/s** for 25 seconds at concurrency 32 – the TTFT gap is the medium supply capability gap.
2. **Concurrency scalability: The saturation inflection points of the two media differ by approximately 4x**. The local disk saturates before concurrency 8 (after which adding concurrency yields zero throughput growth and doubled latency); AISSD5000 supply continues to increase with concurrency, only approaching saturation at concurrency 32. AISSD5000's optimal throughput operating point (concurrency 16, 74.9 tok/s) is 40% higher than the local disk's optimal value.
3. **Compared to no external storage recompute: TTFT is 12.6–20x faster, output throughput is 18x higher** (recompute at concurrency 16: TTFT p50 149.5 s, p90 237 s, throughput 4.1 tok/s) – for long context session recovery of a 480B model, recompute is engineering-unusable; a KV storage layer is a hard requirement.
4. **The more realistic and heavier the workload, the greater the advantage; and supply capability can continue to scale**. In the previous 8K short session test (demand below the local disk ceiling), the two media were tied; in this round of 30K sessions, the advantage fully emerges once demand exceeds that ceiling. This round has already pushed the existing configuration (4 disks + single-port 100GbE) to 91% of its limit (bottleneck at the network port, not the disks); adding ports/disks can scale linearly (device full spec nominal 72 GB/s); the local disk has no room to scale in either bandwidth or capacity (2 TB, already unable to simultaneously accommodate model weights and KV pool).

---

## 2. System Under Test and Environment

### 2.1 Hardware

| Component | Configuration |
|-----------|---------------|
| GPU | 8 × AMD Instinct MI308X (192 GB HBM per card, gfx942), TP=8 single instance + expert parallel |
| CPU / Memory | 2 × AMD EPYC 9654 (384 threads), ~1.5 TB memory |
| **Storage Under Test** | **AISSD5000: 4-disk RAID0 (`/dev/md0`, xfs, 14 TB), NVMe-oF / RoCEv2, single-port 100 GbE** |
| Reference Storage | Local NVMe single disk (`/dev/nvme1n1`, PCIe Gen4, mounted `/srv2`); No external storage (recompute) |

### 2.2 Software

| Component | Version |
|-----------|---------|
| Operating System | Ubuntu 22.04, kernel 6.8.0-124-generic |
| GPU Stack | ROCm 7.2 (gfx942) |
| Inference Engine | vLLM 0.20.1+rocm721 |
| KV Cache Library | LMCache (upstream mainline source compiled 2026-06-29; default async disk load + disk parallel read optimization) |
| Model | Qwen3-Coder-480B-FP8 (MoE, weights ~450 GB, both rounds loaded from AISSD5000 array) |
| Key Parameters | `--tensor-parallel-size 8 --enable-expert-parallel`, `--max-model-len 32768`, `--gpu-memory-utilization 0.9`, `--no-enable-prefix-caching` (cold read verification), `LMCACHE_CHUNK_SIZE=256`, `LMCACHE_MAX_LOCAL_CPU_SIZE=4`, `PYTHONHASHSEED=0` |

### 2.3 Workload and Capacity Baseline

- Single session system prompt prefix ~**29.8K tokens** (reps=530), single session KV ~**7.15 GB** – under TP=8, each of the 8 GPUs holds 1/8 shard, parallel read/write;
- Each round loads **34 distinct sessions** (measured on-disk ~**243 GB**), far exceeding HBM residency and CPU transfer layer (4 GB) capacity;
- Measurement: Single wave N=concurrency (8/16/32), read sessions 0..N-1 once each, decode=64, temperature=0.

---

## 3. Test Methodology

### 3.1 Comparison Design (Three Groups × Concurrency Gradients)

| Group | KV Backend Medium | Concurrency Tiers | iostat Monitoring |
|-------|-------------------|-------------------|-------------------|
| ① AISSD5000 | `/mnt/ws5000/lmcache480` (md0, RAID0) | 8 / 16 / 32 | `/dev/md0` |
| ② Local NVMe | `/srv2/lmcache480_local` (nvme1n1, single disk) | 8 / 16 / 32 | `/dev/nvme1n1` |
| ③ Recompute (No External Storage) | None | 16 | `/dev/md0` (should be ≈0) |

The three groups differ only in KV backend medium; model, weight source, engine parameters, session construction, and concurrency tiers are identical.

### 3.2 Physical Cold Read Guarantee (Five-Fold Control)

1. `--no-enable-prefix-caching`: No prefix KV retained in HBM;
2. LMCache CPU transfer layer compressed to 4 GB: Memory layer cannot accommodate any session;
3. **Before each tier measurement, the host executes `sync; echo 3 > /proc/sys/vm/drop_caches`**: Clears 1.5 TB host memory page cache;
4. Each session read only once: After cache clear, every read is a physical first read;
5. Dual evidence: LMCache need-to-load count (disk group: all requests >0 for all tiers; recompute group: =0) + `iostat` physical read bandwidth matches workload volume (recompute group disk read ≈0 counter-proofs pure recompute).

> Note: The new version of LMCache defaults to async loading; its log `Retrieved ... throughput` numbers represent temporary copy speed (can reach 40+ GB/s), **cannot** be used to determine physical disk reads; physical reads are always based on `iostat`.

### 3.3 Measurement and Collection

- Client measures per-request TTFT (p50/p90/p99/mean) via OpenAI-compatible streaming interface; output throughput = N×64 ÷ total batch time;
- Each tier collects `iostat -x 1` in parallel (peak, busy window mean, busy window duration);
- Between group switches, thoroughly clean up inference processes (including EngineCore/Worker residuals) and confirm HBM returns to zero.

## 4. Test Results

### 4.1 Three-Way Comparison Summary Table

| Tier | TTFT p50 (s) | TTFT p90 (s) | Output Throughput (tok/s) | Disk Peak | Disk Busy Window Avg | Cold Read Verification |
|------|------|------|------|------|------|------|
| **AISSD5000 · Concurrency 8** | **7.53** | 7.54 | **56.6** | 10.26 GB/s | 7.49 GB/s | 8/8 |
| **AISSD5000 · Concurrency 16** | **11.85** | 11.87 | **74.9** | 10.20 GB/s | 8.12 GB/s | 16/16 |
| **AISSD5000 · Concurrency 32** | **26.35** | 26.37 | **71.6** | 10.19 GB/s | **9.29 GB/s (sustained 25s)** | 32/32 |
| Local NVMe · Concurrency 8 | 10.17 | 10.18 | 43.9 | 6.78 GB/s | 5.99 GB/s | 8/8 |
| Local NVMe · Concurrency 16 | 17.31 | 17.32 | 53.6 | 6.78 GB/s | 6.17 GB/s | 16/16 |
| Local NVMe · Concurrency 32 | 35.73 | 35.75 | 53.9 | 6.78 GB/s | 6.42 GB/s | 32/32 |
| Recompute · Concurrency 16 | 149.48 | 237.34 | 4.1 | ≈0 | — | need=0 (pure recompute) |

### 4.2 Core Comparison 1: AISSD5000 vs. Local NVMe (TTFT & Output Throughput)

| Concurrency | Local TTFT p50 | **AISSD5000 TTFT p50** | TTFT Reduction | Local Throughput | **AISSD5000 Throughput** | Throughput Improvement |
|------|------|------|------|------|------|------|
| 8 | 10.17 s | **7.53 s** | **−26%** | 43.9 | **56.6** | **+29%** |
| 16 | 17.31 s | **11.85 s** | **−32%** | 53.6 | **74.9** | **+40%** |
| 32 | 35.73 s | **26.35 s** | **−26%** | 53.9 | **71.6** | **+33%** |

> Interpretation: AISSD5000 wins across all three tiers with consistent margins. The difference stems not from software but from media supply capability — under the same workload demand, the local disk is bottlenecked by its physical limit (6.78 GB/s), causing queuing, while the AISSD5000 sustains supply at 9.3–10.2 GB/s. **The longer the session and the higher the concurrency, the deeper the local disk queue, and the more solid AISSD5000's lead becomes.**

### 4.3 Core Comparison 2: Concurrency Scalability (Saturation Point)

**Disk supply capability vs. concurrency (busy window average):**

| Concurrency | AISSD5000 | Local NVMe |
|------|------|------|
| 8 | 7.49 GB/s | 5.99 GB/s |
| 16 | 8.12 GB/s | 6.17 GB/s |
| 32 | **9.29 GB/s (still rising)** | 6.42 GB/s (pinned at physical ceiling) |

**Output throughput vs. concurrency:**

| Concurrency | AISSD5000 | Local NVMe |
|------|------|------|
| 8 | 56.6 tok/s | 43.9 |
| 16 | **74.9 (optimal operating point)** | 53.6 |
| 32 | 71.6 (slight decline) | 53.9 (plateau: more concurrency yields zero gain) |

Three-tier conclusions:

1. **Local disk saturation occurs before concurrency 8**: bandwidth across the three tiers (5.99→6.17→6.42 GB/s) consistently hugs the 6.78 GB/s physical ceiling; all additional concurrency translates into queuing — from concurrency 16 to 32, throughput is flat (53.6→53.9), while TTFT doubles (17.3→35.7 s).
2. **AISSD5000 saturation appears near concurrency 32**: supply continues to rise with concurrency, reaching 91% of peak at concurrency 32, with only a slight throughput decline. **The saturation point of the two media differs by approximately 4x.**
3. **Optimal operating point**: The local disk solution peaks at ~54 tok/s (maxed out at concurrency 8); the AISSD5000 solution's optimal point is concurrency 16 / **74.9 tok/s (40% higher)**, and even in the overload zone (concurrency 32), it maintains 71.6 tok/s with manageable latency increase. Production recommendation: for this configuration, budget concurrency for long-context recovery at 16–24 for stable AISSD5000 operation; the local disk solution should be throttled above concurrency 8.

### 4.4 Mechanism Verification (Bandwidth Accounting Self-Consistency)

- Concurrency 16 requires moving 16 × 7.15 = **114 GB**: at the AISSD5000 busy window average of 8.12 GB/s, this takes ~14.1 s (measured TTFT p50 11.85 s, batch completion 13.3 s — consistent); at the local disk's 6.17 GB/s, it takes ~18.5 s (measured 17.31 s / 18.8 s — consistent) — **TTFT gap ≈ media bandwidth gap, measurements self-consistent**;
- AISSD5000 peak across three tiers is stable at 10.2 GB/s, >90% of the single-port 100 GbE effective ceiling (~11–12 GB/s) — the supply capability of the current configuration is fully utilized by the real inference workload, and the bottleneck is the network port, not the disks (each of the 4 disks still has headroom);
- The inference software stack is not a bottleneck: TP=8 means each request's KV is read in parallel by 8 GPUs from their respective shards; LMCache's default async loading ensures loading overlap across requests (within the same tier, TTFT p50 ≈ p90, no serial staircasing) — the outcome is entirely determined by the storage media's supply capability.

### 4.5 Core Comparison 3: AISSD5000 vs. No External Storage Recompute (Concurrency 16)

| Metric | Recompute | **AISSD5000** | Benefit |
|------|------|------|------|
| TTFT p50 | 149.48 s | **11.85 s** | **12.6x faster** |
| TTFT p90 | 237.34 s | **11.87 s** | **20.0x faster** |
| Output Throughput | 4.1 tok/s | **74.9 tok/s** | **18.3x higher** |

> Interpretation: Recomputing a 30K prefix for a 480B model under 16-way concurrency yields a TTFT of 2.5–4 minutes, which is practically unusable; KV externalization with reuse compresses this to under 12 seconds. **For long-context large models, "recompute" is not an option; a KV storage layer is a hard requirement**; given that, which medium to choose is answered by §4.2/4.3.

---

## 5. Analysis and Discussion

### 5.1 AISSD5000's Advantage Zone and Boundaries

In previous 8K short-session tests (KV 2 GB per session, concurrency 16, demand ~5.5 GB/s — below the local disk ceiling), the two media were tied. By extending the session to 30K (KV 7.15 GB), the cold recovery demand consistently exceeded the local disk's physical ceiling, and the AISSD5000 won decisively. The pattern is clear: **when demand is below the local disk ceiling, the two are equivalent; when demand exceeds the ceiling, the gap equals the difference in their supply capabilities and widens as load increases.** Production workloads for 480B-class models (long history, multiple sessions, recovery storms) operate precisely in the latter regime.

### 5.2 Structural Differences in Capacity

A single KV pool in this test is 243 GB; the local disk capacity is 2 TB and must share space with model weights and images (during testing, the local disk could not accommodate the 480B weight replica due to space constraints, so the weights were actually hosted on the AISSD5000). A production-grade KV pool has no room on local disks; the AISSD5000 starts at 14 TB, scales by disk, and is inherently shareable across multiple nodes.

### 5.3 Scaling Path

This test already pushed the "4-disk RAID0 + single-port 100 GbE" configuration to the network port ceiling (10.2 GB/s). The AISSD5000 has 6x 100 G ports and 24 drive bays; adding a second port can raise the ceiling to ~20 GB/s, and the full configuration is rated at 72 GB/s — supply capability scales linearly with business growth; the local disk solution has no such path.

---

## 6. Conclusion

1. **Compared to local NVMe SSDs (key conclusion)**: Under the 480B 8-GPU single-instance, long-context cold recovery workload, AISSD5000 reduces **TTFT by 26%–32% and increases output throughput by 29%–40%**, with advantages covering the full concurrency range of 8–32. The mechanism is that local disk bandwidth is pinned at the physical limit of 6.78 GB/s, while AISSD5000 continuously supplies 9.3–10.2 GB/s.
2. **Concurrency scalability**: Local disks saturate before concurrency 8 (adding more concurrency yields zero throughput gain and doubles latency), while AISSD5000 only approaches saturation at concurrency 32—**the saturation inflection point differs by approximately 4×**. The optimal throughput operating point is 40% higher.
3. **Compared to recompute**: TTFT is 12.6–20× faster, throughput is 18× higher. Session recovery for long-context large models must rely on the KV storage layer.
4. **Supply capability is fully utilized and scalable**: Real inference workloads have already pushed the current configuration to 91% of its limit (bottleneck is the single network port, not the drives). Adding ports/drive bays scales linearly (full-config nominal 72 GB/s). Local disk bandwidth and capacity have no room for expansion.

---

## 7. Limitations and Future Work

1. **Single-node scope**: Cross-node sharing of the same KV pool (multi-node aggregation, single prefill reused across the entire cluster) has not been physically verified and is listed as future work.
2. **Capacity wall scenario**: Steady-state throughput comparison when the KV pool exceeds local disk available capacity (forcing local solutions to evict and recompute) is the next experiment to demonstrate capacity advantages.
3. **Recompute group only tested one level** (concurrency 16): Recompute takes too long; one level is sufficient for qualitative analysis.
4. **Synthetic workload**: Uniform access, fixed length, conservative estimate relative to real skewed traffic. Each data point is a single measurement.

---

## Appendix A: Reproduction Commands (Complete for All Three Configurations)

> Convention: Execute directly on the host; `docker exec vllm` to enter the container. Before each group, thoroughly clean up inference processes (including EngineCore/Worker_TP remnants) and confirm HBM is zeroed. After populate completes (blocking foreground), clear cache before measurement.

### A.1 Start Service (① AISSD5000; ②③ Only Change Commented Lines)

```bash
docker exec -d vllm bash -c "export VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO \
 LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
 LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache480 LMCACHE_MAX_LOCAL_DISK_SIZE=1000; \
 vllm serve /mnt/ws5000/models/Qwen3-Coder-480B-FP8 --served-model-name qwen \
 --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
 --max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
 > /mnt/ws5000/kv32k_ws.log 2>&1"
# ② Local disk: LMCACHE_LOCAL_DISK=file:///srv2/lmcache480_local , log kv32k_loc.log
# ③ Recompute: remove LMCACHE_LOCAL_DISK / LMCACHE_MAX_LOCAL_DISK_SIZE , log kv32k_rec.log
```

### A.2 Populate (34 sessions × 29.8K tokens, skip for recompute group)

```bash
docker exec vllm python3 /mnt/ws5000/benchcap_full.py WS32K_pp 530 34 populate 1 1   # ~280s, ~243GB written to disk
```

### A.3 Concurrency Gradient Measurement (clear page cache + iostat before each level)

```bash
for C in 8 16 32; do
  sync; echo 3 > /proc/sys/vm/drop_caches
  nohup bash -c "iostat -x 1 240 /dev/md0 > /tmp/io32k_WS32K_c$C.log 2>&1" &   # ② monitor /dev/nvme1n1
  docker exec vllm python3 /mnt/ws5000/benchcap_off.py WS32K_c$C 530 $C 64 $C 0
  awk '\$1==\"md0\"{if(\$3>m)m=\$3}END{printf \"peak %.2f GB/s\\n\", m/1e6}' /tmp/io32k_WS32K_c$C.log
  awk '\$1==\"md0\" && \$3>500000{c++;s+=\$3}END{if(c)printf \"busy_avg %.2f GB/s (%ds)\\n\", s/c/1e6, c}' /tmp/io32k_WS32K_c$C.log
done
# Verification: grep -acE 'need to load: [1-9]' <service log>   # Disk group should = N, recompute group should = 0
```

## Appendix B: Measurement Script

**`benchcap_off.py`** (Measurement: N sessions single-wave cold read, offset support)

```python
# Usage: python3 benchcap_off.py <label> <reps> <N> <decode> <conc> <offset>
import urllib.request, json, time, sys
from concurrent.futures import ThreadPoolExecutor
BASE='http://localhost:8000/v1/chat/completions'; MODEL='qwen'
label=sys.argv[1]; reps=int(sys.argv[2]); N=int(sys.argv[3]); decode=int(sys.argv[4]); conc=int(sys.argv[5]); off=int(sys.argv[6])
basep='Background: AISSD5000 is a domestic high-performance all-flash NVMe-oF storage, which can serve as a tiered backend medium for KV cache in large model inference, cooperating with vLLM and LMCache to tier KV access across HBM/memory/disk.'
def make_prefix(i): return '[sess-%05d] '%i + basep*reps
def req(sid, maxtok):
    body=json.dumps({'model':MODEL,'stream':True,'messages':[{'role':'system','content':make_prefix(sid)},{'role':'user','content':'Answer%d'%sid}],'max_tokens':maxtok,'temperature':0}).encode()
    st=time.time(); ttft=None; n=0
    try:
        r=urllib.request.urlopen(urllib.request.Request(BASE,data=body,headers={'Content-Type':'application/json'}),timeout=1200)
        for line in r:
            sx=line.decode('utf-8','ignore')
            if sx.startswith('data:') and '"content"' in sx:
                if ttft is None: ttft=time.time()-st
                n+=1
        return (ttft, time.time()-st, n)
    except Exception:
        return (None,None,0)
def pct(a,q): return a[min(len(a)-1,int(len(a)*q))] if a else 0.0
res=[]; t0=time.time()
with ThreadPoolExecutor(max_workers=conc) as ex:
    futs=[ex.submit(req,off+i,decode) for i in range(N)]
    for f in futs:
        x=f.result()
        if x[0] is not None: res.append(x)
wall=time.time()-t0
tt=sorted(r[0] for r in res)
print('[%s] n=%d wall=%.1fs TTFT p50=%.3f p90=%.3f p99=%.3f mean=%.3f'%(label,len(res),wall,pct(tt,.5),pct(tt,.9),pct(tt,.99),(sum(tt)/len(tt) if tt else 0)),flush=True)
```

**`benchcap_full.py`** (populate mode: per-session prefill, KV offload) usage: `python3 benchcap_full.py <label> <reps> <N> populate 1 1`; `reps=530` → ~29.8K tokens per session.

## Appendix C: Raw Data Files

| Path | Content |
|------|---------|
| `/tmp/kv32k.out` | Full experiment orchestration log (TTFT/throughput/bandwidth/forensic output for each tier) |
| `/mnt/ws5000/kv32k_ws.log`, `kv32k_loc.log`, `kv32k_rec.log` | Three-party vLLM/LMCache service logs (need-to-load original text) |
| `/tmp/io32k_WS32K_c{8,16,32}.log`, `/tmp/io32k_LOC32K_c{8,16,32}.log` | iostat physical read time series |
| `/mnt/ws5000/lmcache480`, `/srv2/lmcache480_local` | Two rounds of KV data (~243 GB each) |

---

*All data in this report is from measured results. The test methods, commands, and scripts are fully documented and independently reproducible.*
