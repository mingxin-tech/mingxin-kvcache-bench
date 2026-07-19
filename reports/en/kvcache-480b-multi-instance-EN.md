# FX100 KV-Cache Benchmark (480B, Multi-Instance, Official No.-006)

> **Reference translation.** The signed Chinese original is the authoritative version ([download](https://mingxinstorage.xyz/evidence/R4-kvcache-480b-multi-instance.pdf)). AI-translated by Mingxin's translation pipeline; number fidelity machine-verified (see `dmkt/report_i18n/qc/R4.json`). Report date: 2026-07-06.

## AISSD5000 All-Flash Storage for LLM Inference KV Cache Performance Test Report (480B Model · 8-Card Single Instance & 4-Card Dual Instance · Long Context Cold Recovery)

| Testing Organization | |
|------|------|
| Client | Shenzhen Zhongke Hangxing Technology Co., Ltd. |
| Report No. | AISSD5000-KVC-PERF-2026-006 |
| Issue Date | 2026-07-06 |

## Report Basic Information

| Item | Content |
|------|------|
| Report Title | AISSD5000 All-Flash Storage for LLM Inference KV Cache Tiering Performance Test Report (480B Model · 8-Card Single Instance & 4-Card Dual Instance · Long Context Cold Recovery) |
| Report No. | AISSD5000-KVC-PERF-2026-006 |
| Version | V1.0 |
| Testing Organization | |
| Client | Shenzhen Zhongke Hangxing Technology Co., Ltd. |
| Device Under Test (DUT) | AISSD5000 All-Flash NVMe-oF Storage Array |
| Test Category | Performance Benchmark |
| Classification | Internal |

## Revision History

| Version | Date | Description |
|------|------|------|
| V1.0 | 2026-07-06 | Initial release: Based on Qwen3-Coder-480B-FP8 large model (single-session KV ~7.15 GB, context ~30K tokens), on a single 8-card server in two production deployment modes — 8-card single instance (TP8×1) and 4-card dual instance (TP4×2) — to quantify AISSD5000's long-context cold recovery performance relative to local NVMe and recompute without external storage, and to verify cross-instance KV hot-sharing capability based on the `fs://` shared pool |

---

## 1. Executive Summary

This report, in a controlled single-server (8-card) environment, uses a **480B-parameter large model (Qwen3-Coder-480B-FP8, MoE architecture, weights ~450 GB)** as the workload to quantitatively evaluate the performance value of the AISSD5000 all-flash storage as a KV cache tiering backend for large language model (LLM) inference. Compared to previous tests with 7B/14B models, the single-session KV size for the 480B model increases to approximately **7.15 GB** (context ~30K tokens), representing the real workload pressure of long-history sessions and large models in production environments.

The core feature of this test round is covering two mainstream production deployment modes on the same 8×AMD Instinct MI308X server: **8-card single instance (TP8×1)** — a single large instance, single request read by 8 cards in parallel, pursuing the lowest single-request latency; **4-card dual instance (TP4×2)** — one machine split into two independent medium instances, independent fault tolerance and elastic scaling, a more common multi-instance split in production. The workload is uniformly "long-context session cold recovery" (simulating service restart, session migration, recovery storm of inactive session wake-up). Under completely consistent workload and measurement metrics, three backend media (AISSD5000 / local NVMe / recompute without external storage) are compared with concurrency gradient scanning; the 4-card dual instance mode additionally verifies **cross-instance KV hot-sharing** based on the LMCache built-in `fs://` connector.

**Key Conclusions (all requests successful in each group, disk group physical cold read per request verified):**

1. **Compared to local NVMe (core conclusion): AISSD5000 reduces time to first token (TTFT) by 25%–32% and increases output throughput by 35%–40%, and this benefit holds stably across deployment modes.** For the 8-card single instance mode at concurrency 16, TTFT p50 drops from 17.31 s to **11.85 s (32% reduction)**, output throughput increases from 53.6 to **74.9 tok/s (40% increase)**; for the 4-card dual instance mode at total machine concurrency 16, TTFT p50 drops from 19.03 s to **13.38 s (30% reduction)**, aggregate throughput increases from 49.7 to **66.9 tok/s (35% increase)**. The mechanism is pinned by `iostat` physical read measurements: under the same workload, local disk bandwidth at all concurrency levels is pinned at its physical limit of **6.78 GB/s**, while AISSD5000 peak values at each concurrency level reach **10.2–10.4 GB/s** (90%+ of single-port 100 GbE line rate) and sustain 8.6–9.6 GB/s at high concurrency — the TTFT gap is the media supply capability gap. The benefit does not depend on "how many instances a machine is split into"; it is the capability of the media itself.

2. **Compared to recompute without external storage: TTFT is 8.6–20× faster, output throughput is 17–21× higher, and decode speed is over 75× better.** For the 8-card single instance at concurrency 16, recompute TTFT p50 is 149.5 s, throughput is 4.1 tok/s; for the 4-card dual instance at concurrency 16, recompute TTFT p50 is 114.8 s, and time per output token (TPOT) reaches 1.8 s (normal is only 24 ms, decode is continuously squeezed by prefill, output stream nearly frozen). Repeated prefill of a 30K-long prefix for a 480B-class model is engineering-unusable — **for long-context large model session recovery, "recompute" is not an option; KV storage layer is a hard requirement**; and the smaller the instance compute power is split (TP4 is half of TP8), the more unusable recompute becomes, making the external KV layer a prerequisite for multi-instance deployment.

3. **Concurrency scalability: The saturation inflection point of AISSD5000 and local disk differs by approximately 4×.** Local disk saturates before concurrency 8 (after which adding concurrency yields zero throughput growth and latency doubles); AISSD5000 supply continues to increase with concurrency, approaching saturation only at concurrency 32. For the 8-card single instance mode, AISSD5000's optimal throughput operating point (concurrency 16, 74.9 tok/s) is 40% higher than the local disk optimal value (~54 tok/s). The longer the session and the higher the concurrency, the deeper the local disk queue, and the more solid AISSD5000's lead.

4. **Cross-instance KV hot-sharing is valid with zero performance penalty (4-card dual instance special verification).** Using the LMCache built-in `fs://` shared pool backend, two independent instances mount the same array directory: Instance B cross-cold-reads sessions injected by Instance A, **32/32 full physical hits, zero degraded recompute**. Cross-read is only 3%–8% slower than reading its own, and is still 22% faster than local disk reading its own. This makes "one prefill, full machine reuse", "free session migration across instances", and "instance restart without cache loss" a reality — capabilities built on shared media that local disk solutions semantically cannot provide.

5. **Supply capability is fully utilized by real workload and scales linearly.** This round has already pushed the existing configuration (4-disk RAID0 + single-port 100 GbE) to 90%+ of its limit (bottleneck is the network port, not the disks); AISSD5000 has 6×100 G ports and 24 drive bays, adding ports/drive bays enables linear scaling (full configuration nominal 72 GB/s); local disk solution has no room for expansion in either bandwidth (6.78 GB/s ceiling) or capacity (2 TB, cannot simultaneously accommodate 480B weights and KV pool).

**Applicability Boundary Note:** This test is a performance benchmark under a single 8-card server environment; direct verification of sharing the same KV pool across physical nodes (requires a second host connected to the storage network) is listed as future work.

---

## 2. Test Purpose and Scope

### 2.1 Test Purpose

Answer the following decision questions:

1. Under a 480B large model, long-context (30K token) session cold recovery workload, what TTFT and throughput benefits does AISSD5000 provide for KV Cache compared to local NVMe and recompute without external storage?
2. Do these benefits hold equally under the "8-card single instance" and "4-card dual instance" production deployment modes? How does deployment mode affect the realization of storage value?
3. As concurrency increases, how do the supply capability and saturation behavior of AISSD5000 and local disk evolve?
4. Under multi-instance deployment, can multiple independent instances share the same external KV pool (cross-instance hot-sharing), enabling "one prefill, full machine reuse"?

### 2.2 Test Scope

- **In scope:** Under a single 8-card server environment, three-way media comparison (AISSD5000 / local NVMe / recompute without external storage) for 480B model long-context cold recovery workload; two deployment modes: 8-card single instance (TP8×1) and 4-card dual instance (TP4×2); concurrency gradient scanning (8 / 16 / 32); cross-instance KV hot-sharing verification based on `fs://` shared pool; per-request TTFT (p50/p90/p99), TPOT, aggregate throughput, and physical read bandwidth quantification.
- **Out of scope:** Physical verification of cross-physical-node KV pool sharing, GPUDirect read path (this round uses host-staging asynchronous load path), real business skewed traffic — listed as future work (§9).

---

## 3. Terminology and Metric Definitions

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **KV Cache** | Key-value intermediate results produced during the LLM inference prefill phase; can be reused to skip repeated computation | — |
| **Deployment Mode** | TP8×1: 8 GPUs in tensor parallelism form 1 instance; TP4×2: two groups of 4 GPUs each form an independent instance | — |
| **Long Context Cold Recovery** | Recovery scenario where session KV has been evicted from HBM and memory, requiring physical readback from the storage layer | — |
| **TTFT** (Time to First Token) | Time from request submission to receiving the first output token | Lower is better |
| **TPOT** (Time Per Output Token) | Per-request (end-to-end latency − TTFT) ÷ (number of output tokens − 1) | Lower is better |
| **Aggregate Output Throughput** (tok/s) | Output tokens produced per unit time; for dual-instance, total output tokens of the whole machine ÷ total batch time of the slower instance | Higher is better |
| **p50 / p90 / p99** | 50th/90th/99th percentile of latency (tail latency measures worst-case experience) | Lower is better |
| **Total Machine Concurrency** | Total number of in-flight requests; dual-instance = 2 × per-instance concurrency | — |
| **need-to-load** | Number of tokens that need to be loaded from the lower layer (disk) in LMCache logs; >0 indicates non-GPU/memory hit | — |
| **Physical Read Bandwidth** | Block device read rate measured by `iostat` (page cache cleared before measurement to reflect real disk reads); peak / busy-window average (>0.5 GB/s window) | Higher is better |
| **Cross-Instance Hot Sharing** | KV loaded by one instance can be directly hit and reused by another independent instance | — |
| **Recompute** | Re-executing prefill computation on historical prefixes without external KV storage | Performance lower bound |

> Note: All latency metrics use the OpenAI-compatible streaming interface with temperature=0; the new LMCache defaults to asynchronous loading, and its log `Retrieved ... throughput` is the temporary copy speed, **not** to be used for judging physical disk reads; physical reads are always based on `iostat`.

---

## 4. System Under Test (SUT)

### 4.1 SUT Boundary

The system under test is the end-to-end "inference service + KV cache tiering" system; the controlled variable is the **KV Cache backend medium and deployment mode**. The inference engine, model, sampling parameters, session construction, and concurrency levels are consistent across all groups.

### 4.2 Hardware and Device Under Test (DUT)

| Component | Configuration |
|-----------|---------------|
| GPU | 8 × AMD Instinct MI308X (192 GB HBM per card, gfx942) |
| CPU / Memory | 2 × AMD EPYC 9654 (384 threads), ~1.5 TB memory |
| **Device Under Test (DUT)** | **AISSD5000 all-flash storage, 4-disk RAID0 (`/dev/md0`, xfs, 14 TB), connected via NVMe-oF / RoCEv2, single-port 100 GbE, mounted at `/mnt/ws5000`** |
| Reference Storage | Local NVMe single disk (`/dev/nvme1n1`, PCIe Gen4, mounted at `/srv2`, 2 TB); no backend (recompute) |
| Network | 100 GbE, RoCEv2 |

### 4.3 Software Stack and Versions

| Component | Version |
|-----------|---------|
| Operating System | Ubuntu 22.04, kernel 6.8.0-124-generic |
| GPU Stack | ROCm 7.2 (gfx942) |
| Inference Engine | vLLM 0.20.1+rocm721 |
| KV Cache Library | LMCache (upstream mainline source compiled 2026-06-29, default async disk loading + disk parallel read optimization) |
| Model | Qwen3-Coder-480B-FP8 (MoE, weights ~450 GB, loaded from AISSD5000 array for all rounds) |
| Key Runtime Parameters | `--tensor-parallel-size {8 or 4} --enable-expert-parallel`, `--max-model-len 32768`, `--gpu-memory-utilization 0.9`, `--no-enable-prefix-caching` (forces cold read verification), `LMCACHE_CHUNK_SIZE=256`, `PYTHONHASHSEED=0` |

### 4.4 Workload and Capacity Baseline

- Model is Qwen3-Coder-480B-FP8 (MoE architecture, expert parallelism EP); single-session system prompt prefix is ~**29.8K tokens** (reps=530), **single-session KV is ~7.15 GB** — approximately 4.6 times that of a 14B model (1.55 GB/8.4K token session), with load pressure close to production-scale large-model long-history session scenarios.
- For the 8-GPU single-instance mode, each round loads **34 distinct sessions** (measured on-disk ~243 GB); for the 4-GPU dual-instance mode, two instances load **48 distinct sessions** in parallel (A loads 0–23, B loads 24–47, on-disk ~343 GB). The working set far exceeds HBM residency and CPU staging layer capacity, ensuring measured sessions necessarily overflow to the storage layer.
- Measurement: single wave N=concurrency, each session read once, decode=64, temperature=0. 8-GPU single-instance concurrency levels: 8/16/32; 4-GPU dual-instance total machine concurrency levels: 16 (8+8) / 32 (16+16), aligned with single-instance concurrency.

### 4.5 Deployment Mode Description

- **8-GPU Single Instance (TP8×1)**: 8 GPUs in tensor parallelism + expert parallelism form a single 480B service instance; a single request's KV is read/written in parallel by 8 GPUs each holding 1/8 shard. Pursues the lowest single-request recovery latency, representing a "single large service" deployment.
- **4-GPU Dual Instance (TP4×2)**: GPUs 0–3 form instance A (port 8000), GPUs 4–7 form instance B (port 8001), each an independent 480B service, each with 4-GPU TP + EP. Represents the more common production "multi-service per machine" split (higher instance-level fault tolerance, independent scaling, single-instance failure does not bring down the whole machine). The two instances start staggered, load weights from the array in parallel (total ~900 GB), and are both ready in 4.5 minutes.

---

## 5. Test Methodology

### 5.1 Test Design

**Experiment 1 (8-GPU Single Instance TP8×1) — Three groups × concurrency gradient:**

| Group | KV Backend Medium | Concurrency Levels | iostat Monitoring |
|-------|-------------------|--------------------|-------------------|
| ① AISSD5000 | `/mnt/ws5000/lmcache480` (md0, RAID0) | 8 / 16 / 32 | `/dev/md0` |
| ② Local NVMe | `/srv2/lmcache480_local` (nvme1n1, single disk) | 8 / 16 / 32 | `/dev/nvme1n1` |
| ③ Recompute (no external storage) | None | 16 | `/dev/md0` (should be ≈0) |

**Experiment 2 (4-GPU Dual Instance TP4×2) — Four groups × two total machine concurrency levels:**

| Group | KV Backend Configuration | Total Machine Concurrency | Description |
|-------|--------------------------|---------------------------|-------------|
| ④ AISSD5000 · Independent Pool | `LMCACHE_LOCAL_DISK` (md0) | 16 / 32 | LocalDiskBackend, per-instance process index |
| ⑤ AISSD5000 · fs Shared Pool | `LMCACHE_REMOTE_URL=fs://…` (md0) | 16 / 32 (includes cross-reads) | FSConnector, index is filesystem, dual instances share one pool |
| ⑥ Local NVMe | `LMCACHE_LOCAL_DISK` (nvme1n1) | 16 / 32 | Local single disk |
| ⑦ Recompute (no external storage) | LMCache not mounted | 16 / 32 | Pure GPU recompute baseline |

> The only difference between groups is the KV backing medium/configuration; deployment form, model, weight source, engine parameters, session construction, and concurrency levels are identical. The `fs://` shared pool URL must be written in the placeholder form `fs://local:0/path` — in this version, `parse_remote_url()` enforces host:port validation; writing `fs:///path` per the official documentation causes the remote backend connection to fail and KV to be silently dropped (this was discovered and avoided in this round).

### 5.2 Physical Cold Read Guarantee (Multiple Controls)

1. `--no-enable-prefix-caching`: No prefix KV is retained in HBM;
2. LMCache CPU transfer layer limited to 4 GB (far below the 7.15 GB per session): The memory layer cannot hold any session;
3. **Before each level measurement, the host executes `sync; echo 3 > /proc/sys/vm/drop_caches`**: Clears the 1.5 TB host memory page cache;
4. Each session is read only once (in the dual-instance form, the read sets of the two instances are disjoint, with no cross-instance page cache sharing);
5. Dual verification: The disk group has full hits per request (`hit tokens ≈ 30208`) and `need to load > 0`, with `iostat` physical read volume matching the working set; the recompute group has no LMCache process and disk reads ≈ 0.

### 5.3 Measurement and Data Collection

- The client measures per-request TTFT (p50/p90/p99/mean) and TPOT via the OpenAI-compatible streaming interface; aggregate throughput = total output tokens ÷ total batch time (for dual instances, the slower instance's total batch time is used);
- `iostat -x 1` is collected in parallel for each level (peak, busy-window mean, busy-window duration);
- Between group switches, inference processes (including EngineCore/Worker remnants) are thoroughly cleaned, and 8-card HBM is confirmed to be zeroed.

### 5.4 Methodological Controls (Preventing Data Contamination)

1. **Forced Physical Read Proof**: All requests in the disk group have full hits and need>0, with no GPU/memory hits; the recompute group has need=0 and disk reads ≈ 0, proving pure recompute;
2. **Single Variable Verification**: At each startup, environment variables (disk directory / shared URL / TP degree) and LMCache configuration log echoes are verified;
3. **Reproducibility**: All metrics for the four-card dual-instance configuration were obtained from two independent complete measurements on the same day, with deviations ≤5% per level;
4. **Correctness**: All requests in each group returned successfully; with the same prompt, temperature=0, and same decode limit, output token counts are consistent.

---

## 6. Test Results

### 6.1 Experiment 1: Eight-Card Single Instance (TP8×1) Three-Way Comparison Summary Table

Measurement conditions: 480B model, ~30K token / KV 7.15 GB per session, single wave N=concurrency, decode=64, disk group page cache cleared, physical hit per request.

| Level | TTFT p50 (s) | TTFT p90 (s) | Output Throughput (tok/s) | Disk Peak (GB/s) | Disk Busy-Window Mean (GB/s) | Cold Read Proof |
|------|------|------|------|------|------|------|
| AISSD5000·Concurrency 8 | **7.53** | 7.54 | **56.6** | 10.26 | 7.49 | 8/8 |
| AISSD5000·Concurrency 16 | **11.85** | 11.87 | **74.9** | 10.20 | 8.12 | 16/16 |
| AISSD5000·Concurrency 32 | **26.35** | 26.37 | **71.6** | 10.19 | 9.29 (sustained 25s) | 32/32 |
| Local NVMe·Concurrency 8 | 10.17 | 10.18 | 43.9 | 6.78 | 5.99 | 8/8 |
| Local NVMe·Concurrency 16 | 17.31 | 17.32 | 53.6 | 6.78 | 6.17 | 16/16 |
| Local NVMe·Concurrency 32 | 35.73 | 35.75 | 53.9 | 6.78 | 6.42 | 32/32 |
| Recompute·Concurrency 16 | 149.48 | 237.34 | 4.1 | ≈0 | — | need=0 (pure recompute) |

### 6.2 Experiment 2: Four-Card Dual Instance (TP4×2) Full Metrics Summary Table

Measurement conditions: 480B model, ~30K token / KV 7.15 GB per session, both instances launched simultaneously, full-machine scope (A+B combined samples), decode=64, disk group page cache cleared, physical hit per request.

| Level | Full-Machine Concurrency | TTFT p50 (s) | TTFT p90 (s) | TTFT p99 (s) | TPOT p50 (ms) | Aggregate Throughput (tok/s) | Disk Peak (GB/s) | Disk Busy-Window Mean (GB/s) |
|------|------|------|------|------|------|------|------|------|
| AISSD5000·Independent Pool·16 | 16 | **13.38** | 13.78 | 13.79 | 24.2 | **66.9** | 10.32 | 8.56 |
| AISSD5000·Independent Pool·32 | 32 | **25.62** | 26.24 | 26.25 | 32.0 | **72.6** | 10.41 | 9.59 |
| AISSD5000·fs Shared·Read Self·32 | 32 | 24.83 | 26.03 | 26.04 | 30.7 | 73.4 | 10.36 | 9.36 |
| AISSD5000·fs Shared·Cross Read·32 | 32 | 26.81 | 26.83 | 26.84 | 31.6 | 71.1 | 10.38 | 9.22 |
| Local NVMe·16 | 16 | 19.03 | 19.04 | 19.04 | 24.1 | 49.7 | 6.79 | 6.31 |
| Local NVMe·32 | 32 | 34.48 | 36.44 | 36.44 | 30.9 | 53.3 | 6.78 | 6.45 |
| Recompute·16 | 16 | 114.78 | 210.41 | 268.52 | 1817.4 | 3.8 | ≈0 | — |
| Recompute·32 | 32 | 251.69 | 466.95 | 608.52 | 5244.0 | 3.4 | ≈0 | — |

> Cross read = Instance A reads sessions injected by Instance B, Instance B reads sessions injected by Instance A (32/32 full physical hits); Read self = each reads its own injected sessions.

### 6.3 Injection Phase Data

| Form / Group | Sessions Injected | Duration (s) | Data Written |
|----|--------|----------|--------|
| Eight-Card Single Instance·AISSD5000 | 34 | ~200 | AISSD5000 ~243 GB |
| Eight-Card Single Instance·Local Disk | 34 | ~200 | Local Disk ~243 GB |
| Four-Card Dual Instance·AISSD5000 (Parallel Inject) | 48 (A24+B24) | 361–365 | AISSD5000 ~343 GB |
| Four-Card Dual Instance·Local Disk (Parallel Inject) | 48 | 358–363 | Local Disk ~343 GB |
| Four-Card Dual Instance·fs Shared Pool (Parallel Inject) | 48 | 363 | AISSD5000 ~344 GB / 22656 files |
| Recompute Group | — (no injection needed) | — | 0 |

> Dual-instance parallel injection of 343 GB took only 363 s (~0.95 GB/s sustained write + dual-engine prefill read-write mix); injection times for the WS round and local round are nearly identical — write demand is far below the upper limit of both media, injection is compute-bound and does not affect comparison fairness.

### 6.4 Core Comparison 1: AISSD5000 vs Local NVMe (Two Deployment Forms)

**Eight-Card Single Instance (TP8×1):**

| Concurrency | Local TTFT p50 | AISSD5000 TTFT p50 | TTFT Reduction | Local Throughput | AISSD5000 Throughput | Throughput Improvement |
|------|------|------|------|------|------|------|
| 8 | 10.17 s | **7.53 s** | **−26%** | 43.9 | **56.6** | **+29%** |
| 16 | 17.31 s | **11.85 s** | **−32%** | 53.6 | **74.9** | **+40%** |
| 32 | 35.73 s | **26.35 s** | **−26%** | 53.9 | **71.6** | **+33%** |

**Four-GPU Dual Instance (TP4×2, Full-Machine Perspective):**

| Full-Machine Concurrency | Local TTFT p50 | AISSD5000 TTFT p50 | TTFT Reduction | Local Aggregate Throughput | AISSD5000 Aggregate Throughput | Throughput Improvement |
|------|------|------|------|------|------|------|
| 16 | 19.03 s | **13.38 s** | **−30%** | 49.7 | **66.9** | **+35%** |
| 32 | 34.48 s | **25.62 s** | **−26%** | 53.3 | **72.6** | **+36%** |

> Interpretation: The conclusions are highly consistent across both deployment forms (TTFT reduced by 26%–32%, throughput increased by 29%–40%) — **the benefit of AISSD5000 does not depend on "how many instances are split on one machine"; it is a difference in media supply capability**. Under the same load demand, all tiers of physical read bandwidth on local drives are pinned at the 6.78 GB/s physical ceiling (busy-window average 5.99–6.64 GB/s), while AISSD5000 supplies 8.6–9.6 GB/s continuously, peaking at 10.2–10.4 GB/s (90%+ of the single-port 100 GbE line rate). The longer the session and the higher the concurrency, the deeper the queue on local drives, and the more solid AISSD5000's lead becomes.

### 6.5 Core Comparison 2: AISSD5000 vs. No External Storage Recomputation

| Form / Concurrency | Metric | Recomputation | AISSD5000 | Benefit |
|------|------|------|------|------|
| Eight-GPU Single Instance·16 | TTFT p50 | 149.48 s | **11.85 s** | **12.6x faster** |
| Eight-GPU Single Instance·16 | TTFT p90 | 237.34 s | **11.87 s** | **20.0x faster** |
| Eight-GPU Single Instance·16 | Output Throughput | 4.1 tok/s | **74.9 tok/s** | **18.3x higher** |
| Four-GPU Dual Instance·16 | TTFT p50 | 114.78 s | **13.38 s** | **8.6x faster** |
| Four-GPU Dual Instance·16 | TTFT p99 | 268.52 s | **13.79 s** | **19.5x faster** |
| Four-GPU Dual Instance·16 | TPOT p50 | 1817 ms | **24.2 ms** | **75x better** |
| Four-GPU Dual Instance·16 | Aggregate Throughput | 3.8 tok/s | **66.9 tok/s** | **17.6x higher** |
| Four-GPU Dual Instance·32 | TPOT p50 | 5244 ms | **32.0 ms** | **164x better** |

> Interpretation: For the 480B model recomputing a 30K prefix, not only is the first token an order of magnitude slower, but **decoding is also crippled** — continuous prefill squeezes the decode batch, causing each token decode interval (TPOT) to reach 1.8–5.2 seconds (normal 24–32 ms), making the output stream after the first token equally unusable. In the four-GPU dual-instance form, the TP4 instance has half the compute power of TP8, and the recomputation degradation is deeper — **the finer the instance is split, the more essential the KV external tier becomes**. For session recovery in long-context, ultra-large models, "recomputation" is not an option.

### 6.6 Core Comparison 3: Concurrency Scalability (Saturation Knee)

**Drive supply capability vs. concurrency (eight-GPU single instance, busy-window average):**

| Concurrency | AISSD5000 | Local NVMe |
|------|------|------|
| 8 | 7.49 GB/s | 5.99 GB/s |
| 16 | 8.12 GB/s | 6.17 GB/s |
| 32 | **9.29 GB/s (still rising)** | 6.42 GB/s (pinned at physical ceiling) |

Three conclusions:

1. **The saturation knee for local drives is reached before concurrency 8**: The three bandwidth tiers (5.99→6.17→6.42 GB/s) always hug the 6.78 GB/s physical ceiling; any further concurrency increase is entirely converted into queuing — concurrency 16→32 yields zero throughput growth (53.6→53.9), while TTFT doubles (17.3→35.7 s).
2. **The saturation knee for AISSD5000 only appears around concurrency 32**: Supply continues to rise with concurrency, reaching 91% of its peak at concurrency 32, with only a slight throughput drop. **The saturation knee of the two media differs by approximately 4x.**
3. **Optimal operating point**: The local drive solution achieves a best throughput of ~54 tok/s (peaked at concurrency 8); the AISSD5000 solution's optimal operating point is concurrency 16 / **74.9 tok/s (40% higher)**. Production recommendation: Under this configuration, budget concurrency for long-context recovery at 16–24; AISSD5000 can handle it stably. The local drive solution should be rate-limited above concurrency 8.

### 6.7 Core Comparison 4: Cross-Instance KV Hot Sharing (Four-GPU Dual Instance fs:// Shared Pool Special)

Two instances mount the same array shared pool, cross-cold-reading sessions injected by the other, compared with reading their own under the same metric (full-machine concurrency 32 tier):

| Comparison | TTFT p50 | TTFT p99 | TPOT p50 | Aggregate Throughput | Drive Busy-Window Average | Hit Verification |
|------|------|------|------|------|------|------|
| Independent Pool·Read Self | 25.62 s | 26.25 s | 32.0 ms | 72.6 tok/s | 9.59 GB/s | 32/32 full hits |
| fs Shared Pool·Read Self | 24.83 s | 26.04 s | 30.7 ms | 73.4 tok/s | 9.36 GB/s | 32/32 full hits |
| **fs Shared Pool·Cross-Read Other** | **26.81 s** | **26.84 s** | **31.6 ms** | **71.1 tok/s** | 9.22 GB/s | **32/32 full hits, zero misses** |

> Interpretation: Three conclusions. First, **zero performance cost for shared backend** — reading self from the fs shared pool shows all metrics within noise of the independent backend. Second, **near-zero cost for cross-instance sharing** — cross-read is only 8% slower (p50) / 3% slower (p99) than reading self; all 32 cross-read requests achieved full physical hits (hit tokens per request = 30208), with zero degraded recomputation. Third, **cross-instance reading what the other injected is still 22% faster than local drives reading their own** (26.81 vs 34.48 s) — shared semantics combined with media advantage. In contrast, the LocalDiskBackend's cross-instance attempt (in-process index, cross-read hit=0, degraded recomputation, TTFT 67 s) shows that sharing capability is determined by backend choice; `fs://` achieves it in one step.

> Engineering note: The CPU relay layer for the `fs://` remote path must be ≥ the in-flight KV volume at concurrency. The 4 GB relay layer in this round triggered "allocation failure → blocking timeout cancel retry" at low concurrency, causing individual request decode divergence (identified as a configuration issue after 5 rounds of retesting, not a coincidence). After raising `LMCACHE_MAX_LOCAL_CPU_SIZE` to 64 GB, alerts dropped to zero, and multiple consecutive true cold reads were all clean. Production recommendation: configure the relay layer to be ≥ the in-flight KV volume at concurrency (≥57 GB for this load).

---

## 7. Analysis and Discussion

### 7.1 Robustness of Benefits: Consistent Across Deployment Forms

Two deployment modes (TP8×1, TP4×2) were run independently on the same machine under the same load specification. The performance gains of AISSD5000 over local NVMe are highly consistent (TTFT −26% to −32%, throughput +29% to +40%). The mechanism is the same: the instantaneous bandwidth demand of a 30K long-context cold recovery exceeds the physical ceiling of the local disk (6.78 GB/s), and the gap is exactly the difference in supply capability between the two media. **This conclusion is insensitive to deployment mode and instance granularity and can be directly extrapolated to production partitioning of "one machine, N instances."**

### 7.2 Impact of Deployment Mode on Storage Supply

In both modes, the AISSD5000 peak is 10.2–10.4 GB/s (90%+ of single-port line rate), while the local disk is pinned at 6.78 GB/s—the array can be fully saturated either by the 8-way parallel reads of a single TP8 instance or by the concurrent reads of two independent TP4 instances, and the supply to the two instances is balanced (TTFT p50 difference between instances ≤1.8 s). Under the same concurrency, a single request in a TP8 instance recovers slightly faster (each request is read in parallel by 8 GPUs across shards); at concurrency 32, the two modes nearly overlap. **Mode selection can be based entirely on business needs (fault tolerance/elasticity vs. single-request latency); storage imposes no constraint.**

### 7.3 Mode Sensitivity of Recomputation

KV recovery from TP8 to TP4×2 is only about 2%–14% slower, while recomputation at concurrency 16 goes from 149.5 s (TP8) to 114.8 s per instance (TP4×2) (at concurrency 32, it reaches 251.7 s, TPOT 5.2 s)—recomputation time degrades significantly as instance compute power is partitioned. The production trend is toward more instances/smaller instances (for fault tolerance and elasticity), which turns the KV externalization layer from an "optimization" into a "deployment prerequisite."

### 7.4 Architectural Value of Cross-Instance Hot Sharing

The core advantage of the `fs://` shared pool is that its index is the filesystem itself (`exists()` directly stats the file, with no in-process state), making it natively visible across instances and restarts. Writes use "temporary file + atomic rename," so readers never see a half-written file. This yields three capabilities that local disk schemes cannot provide semantically: **prefill cost is paid only once per machine** (any long prefix/historical session processed by any instance is immediately hit by others); **sessions can be freely migrated across instances** (load balancing requires no session affinity constraint; any instance can recover in 13–27 s instead of 115–252 s via recomputation); **rolling instance restarts do not clear the cache** (the pool resides on the array, and the index is the filesystem).

### 7.5 Supply Capacity Expansion Path

This round already pushed the "4-disk RAID0 + single-port 100 GbE" configuration to the network port ceiling (10.2–10.4 GB/s, bottleneck at the network port, not the disks; each of the 4 disks still has headroom). The AISSD5000 has 6 × 100 G ports and 24 drive bays; adding a second port raises the ceiling to approximately 20 GB/s, with a full-configuration nominal rating of 72 GB/s—supply capacity scales linearly with business growth. The local disk scheme has no room for expansion in either bandwidth (6.78 GB/s ceiling) or capacity (2 TB; during testing, it could no longer simultaneously hold the 480B weights and the KV pool—the weights were actually served by the AISSD5000).

---

## 8. Conclusion

1. **Compared to local NVMe (core conclusion)**: Under a 480B long-context cold recovery load, the AISSD5000 reduces **TTFT by 26%–32% and increases output throughput by 29%–40%**, and this benefit holds in both the 8-GPU single-instance and 4-GPU dual-instance deployment modes. The mechanism is that the local disk is pinned at a 6.78 GB/s physical ceiling, while the AISSD5000 delivers a sustained 8.6–9.6 GB/s (90%+ of single-port line rate).
2. **Compared to no external storage recomputation**: TTFT is 8.6–20× faster, output throughput is 17–21× higher, and decode speed is >75× better; recomputation is unusable in both TTFT and decode phases. The KV externalization layer is a prerequisite infrastructure for multi-instance deployment of long-context, very large models, and this conclusion strengthens as instances are partitioned finer.
3. **Concurrency scalability**: The local disk saturates before concurrency 8 (beyond which concurrency yields zero throughput gain and doubles latency), while the AISSD5000 only approaches saturation at concurrency 32—**the saturation inflection point differs by approximately 4×**; the optimal throughput operating point is 40% higher.
4. **Cross-instance hot sharing**: Based on the `fs://` shared pool, two independent instances cross-reading 32/32 achieve full hits with zero performance penalty (cross-read is only 3%–8% slower than self-read, yet still 22% faster than local disk self-read), enabling one prefill to be reused across the entire machine, free session migration, and cache persistence across restarts.
5. **Supply capacity is fully utilized and scalable**: Real inference load already pushes the current configuration to 90%+ of its ceiling (bottleneck is the single network port, not the disks); adding ports/drive bays enables linear scaling (full-configuration nominal 72 GB/s). The local disk scheme has no room for expansion in either bandwidth or capacity.

---

## 9. Limitations and Future Work

1. **Single-machine scope**: Direct physical verification of sharing a single KV pool across physical nodes (multi-machine aggregation, one prefill reused across the entire cluster) requires a second host connected to the storage network and is planned as future work. Cross-node use of the `fs://` shared pool requires a shared filesystem layer (NFS/cluster filesystem); direct block device attachment to multiple hosts is not feasible.
2. **Access path**: This round used the host-staging asynchronous load path (default in the new LMCache version) and did not cover the GPUDirect read path on this platform.
3. **Synthetic load and statistical repetition**: Uniform access and fixed-length synthetic data represent a conservative estimate relative to real skewed traffic. Key points have been measured twice independently (deviation ≤5%); expanding the sample to provide confidence intervals is recommended.
4. **Uncovered capabilities**: Dual 100 G port aggregation, longer contexts (>32K), and steady-state throughput of cross-restart KV persistence reuse are listed as future test items.

---

## Appendix A: Reproduction Commands (Colleague Server, Inside Container `vllm`)

### A.1 Eight-GPU Single Instance (TP8×1) Three-Way Comparison (Local disk round: change `LMCACHE_LOCAL_DISK` to point to `/srv2/...`; recompute round: remove LMCACHE variables and `--kv-transfer-config`)

```bash
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8
docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 \
 LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
 LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache480 LMCACHE_MAX_LOCAL_DISK_SIZE=1000; \
 vllm serve \$MODEL --served-model-name qwen \
 --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
 --max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
 --port 8000 > /mnt/ws5000/vllm_ws.log 2>&1"
## Inject 34 sessions (reps=530≈30K token) → clear page cache → measure at concurrency 8/16/32 (see A.3)
```

### A.2 Four-GPU Dual Instance (TP4×2) Startup (Independent Pool / fs Shared Pool / Local Disk / Recompute, Only Storage Environment Variables Differ)

## Independent pool:  ENVS="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
##                LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache480tp4 LMCACHE_MAX_LOCAL_DISK_SIZE=1000"
## fs shared pool: ENVS="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=64 \
##                LMCACHE_REMOTE_URL=fs://local:0/mnt/ws5000/kvpool_fs LMCACHE_REMOTE_SERDE=naive"
## Local disk:  Same as independent pool but LMCACHE_LOCAL_DISK=file:///srv2/lmcache480tp4_local
## Recompute:    ENVS is empty and --kv-transfer-config is removed
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

### A.3 Populate, Measure, and Cold Recovery Verification

```bash
## Populate (dual instance parallel; single instance only populates 34 sessions on port 8000)
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 populate 530 24 0  > /mnt/ws5000/ppA.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 populate 530 24 24 > /mnt/ws5000/ppB.log 2>&1"
## Per-tier measurement (example: full-machine concurrency 32 = 16+16; cross-read tier A/B start offset swapped to 24 / 0)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
iostat -x 1 400 /dev/md0 > /tmp/io.log &        # Local disk group monitoring /dev/nvme1n1
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 measure 530 16 0  64 16 > /mnt/ws5000/A.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 16 24 64 16 > /mnt/ws5000/B.log 2>&1"
## Cold recovery verification (each instance measurement requests should all be full disk hits)
grep -acE 'hit tokens: 30[0-9]{3}' /mnt/ws5000/log_i0.log
awk '$1=="md0"{if($3>m)m=$3} $1=="md0" && $3>500000{c++;s+=$3} END{printf "Peak %.2f Busy-avg %.2f GB/s (%ds)\n", m/1e6, s/c/1e6, c}' /tmp/io.log
```

## Appendix B: Load Client `bench_mp.py` (Port-Parameterized, Per-Request TTFT/TPOT)

Usage: `python3 bench_mp.py <port> <mode:populate|measure> <reps> <N> <off> [decode] [conc]`; `reps=530` → approximately 30K tokens per session. In the four-GPU dual-instance configuration, the two instances target ports 8000/8001 respectively; full-machine metrics are computed by merging per-request samples from both instances.

```python
import urllib.request, json, time, sys
from concurrent.futures import ThreadPoolExecutor
port=sys.argv[1]; mode=sys.argv[2]; reps=int(sys.argv[3]); N=int(sys.argv[4]); off=int(sys.argv[5])
decode=int(sys.argv[6]) if len(sys.argv)>6 else 64
conc=int(sys.argv[7]) if len(sys.argv)>7 else 16
BASE='http://127.0.0.1:%s/v1/chat/completions'%port
basep='Background: AISSD5000 is a domestic high-performance all-flash NVMe-oF storage, which can serve as a tiered backend medium for large model inference KV cache, cooperating with vLLM and LMCache to tier KV access between HBM/memory/disk.'
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
tt=sorted(r[1] for r in res); tpots=sorted((r[2]-r[1])/(r[3]-1)*1000 for r in res if r[3]>1); tot=sum(r[3] for r in res)
print('[p%s] n=%d wall=%.1fs TTFT p50=%.2f p90=%.2f p99=%.2f mean=%.2f | TPOT_ms p50=%.1f mean=%.1f | outtok/s=%.1f'%(
  port,len(res),wall,pct(tt,.5),pct(tt,.9),pct(tt,.99),sum(tt)/len(tt),pct(tpots,.5),sum(tpots)/len(tpots),tot/wall),flush=True)
```

## Appendix C: Raw Data Files (Colleague Server)

| File | Content |
|------|---------|
| `/tmp/kv32k.out` | Full process log of three-way comparison for eight-card single instance (TP8×1) |
| `/tmp/tp4x2b.out`, `/tmp/tp4x2c.out`, `/tmp/full2.out` | Full metric measurement orchestration logs for four-card dual instance (TP4×2) |
| `/tmp/fsshare.out`, `/tmp/fsrep*.out` | fs shared pool cross-instance hot sharing validation and divergence retest logs |
| `/mnt/ws5000/results/*_{A,B}.log` | Per-request raw data (REQ lines) and summary lines for each configuration of two instances |
| `/mnt/ws5000/*_i{0,1}.log`, `/mnt/ws5000/vllm_ws.log` | Complete logs for each service instance (including per-request hit evidence lines, FS connector initialization lines) |
| `/tmp/io*_*.log` | Per-second time series of iostat physical read bandwidth for each configuration |
| `/mnt/ws5000/kvpool_fs/` | fs shared pool (344 GB / 22656 files, retained for re-examination) |

---

*All data in this report are from measured results. The test methods, commands, and scripts are fully archived and independently reproducible.*
