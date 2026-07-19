# FX100 KV-Cache Cross-Instance Hot-Sharing Verification (fs:// Shared Pool, Official)

> **Reference translation.** The signed Chinese original is the authoritative version. AI-translated by Mingxin's translation pipeline; number fidelity machine-verified (see `dmkt/report_i18n/qc/X2.json`). Report date: 2026-07-06.

## AISSD5000 KV Cache Cross-Instance Hot Sharing Validation Report — fs:// Shared Pool · 480B · TP4×2

**Test Platform**: AMD Instinct MI308X ×8 (192 GB HBM per card) / ROCm 7.2 / vLLM 0.20.1+rocm721 + LMCache v1 (upstream mainline source compiled)
**Storage Under Test**: AISSD5000 (WS5000) all-flash NVMe-oF array (4-disk RAID0, RoCEv2 over 100GbE, XFS, 14 TB)
**Model**: Qwen3-Coder-480B-FP8 (MoE, weights ~450 GB)
**Deployment**: Dual instance TP4×2 (Instance A: cards 0–3 / port 8000; Instance B: cards 4–7 / port 8001)
**Workload**: Long context cold recovery — ~29.8K tokens per session, KV ≈ 7.15 GB/session
**Date**: 2026-07-06
**Related Report**: *AISSD5000 KV Cache Performance Test Report (480B · TP4×2 Dual Instance · Official)* §4.6 — Closed-loop validation of outstanding issue

---

## 1. Background and Objective

The TP4×2 dual-instance test (2026-07-06 morning) revealed: LMCache `LocalDiskBackend` maintains its key index in process memory, so Instance B could not hit KV prefilled by Instance A into the shared array (hit=0, degraded to recompute, TTFT 67 s) — a software-level gap existed for cross-instance hot sharing.

Solution analysis identified the fix path: switch to LMCache's built-in **`fs://` remote connector** (`FSConnector`). Its index is the filesystem itself (`contains()` directly stats the file), with no in-process state, theoretically supporting cross-instance, cross-restart sharing natively. This round's objectives:

1. **Feasibility verification**: Confirm the fs connector's configuration wiring, registration, and metadata integrity in this version of the code;
2. **Functional validation**: Both instances mount the same shared pool, cross-cold-read each other's prefilled sessions, verify 100% hit rate;
3. **Performance validation**: Compare directly with `LocalDiskBackend` (same machine, same workload baseline from the morning) to confirm sharing does not come at a performance cost.

**Summary of findings: The solution is fully validated.**

1. **Cross-instance hot sharing achieved**: Instances A and B simultaneously cross-cold-read 32 sessions prefilled by the other instance (229 GB total), **32/32 full disk hits, zero misses**, TTFT p50 27.1–27.4 s, essentially no difference from reading their own prefilled sessions (25.0–26.7 s) — **sharing overhead ~5%, near zero**;
2. **Performance on par with LocalDiskBackend**: Under the same measurement scope (full-machine concurrency 16/32), TTFT, aggregate throughput, and array supply are all identical (10.4 GB/s peak, 9.0–9.2 GB/s busy-window average) — sharing capability does not come at a performance cost;
3. **One copy of KV, reusable across the entire machine, becomes reality**: Any long context prefilled by one instance is immediately available to other instances — sessions can be freely migrated/load-balanced between instances, prefilling cost is paid only once across the entire machine;
4. **Discovered and worked around an upstream bug**: The documentation example `fs:///path` fails the URL validation in this version (which mandates host:port), requiring the placeholder form `fs://local:0/path` (reproduction details attached in this report, can be filed as an upstream issue).

---

## 2. Feasibility Verification (Code Level)

Verification results on the LMCache source code (`/root/LMCache/lmcache`, upstream mainline 2026-06-29) inside the colleague's server container:

| Verification Item | Location | Conclusion |
|------|------|------|
| Configuration wiring | `v1/config.py` L114–119 | `remote_url`/`remote_serde` supports environment variables `LMCACHE_REMOTE_URL`/`LMCACHE_REMOTE_SERDE` (marked deprecated but fully functional) |
| Connector registration | `connector/fs_adapter.py` | `FsConnectorAdapter` matches `fs://` prefix, extracts directory path from URL |
| Index mechanism | `connector/fs_connector.py` `exists()` | Direct `os.path.exists(file)` — **index is the filesystem, no in-process state** |
| Tail chunk metadata | `connector/base_connector.py` L55–58 | `save_chunk_meta` defaults to True: each file carries a 28-byte header (shape/dtype/fmt), tail chunks under 256 tokens can be losslessly restored |
| Write atomicity | `fs_connector.py` `put()` | Temporary file write + `os.replace()` atomic rename — reader never sees a half-written file |
| Landing confirmation | Instance logs | All 4 workers show `Creating FS connector` + `Connection initialized at fs://local:0/mnt/ws5000/kvpool_fs` |

**Discovered upstream bug**: `parse_remote_url()` (`connector/__init__.py` L67–68) enforces `assert host/port non-empty` for all URLs, while the documentation example gives `fs:///tmp/lmcache` without a host — configuring per the documentation causes RemoteBackend to retry connection every 30 s and fail permanently (KV silently dropped, pool directory zero files, only visible in WARNING logs). Confirmed workaround via container unit test: **`fs://local:0/mnt/ws5000/kvpool_fs`** (placeholder host:port, fs adapter only uses the path) parses successfully.

---

## 3. Experimental Design

### 3.1 Configuration (only storage backend differs from the morning TP4×2 baseline)

```
LMCACHE_CHUNK_SIZE=256  LMCACHE_LOCAL_CPU=True  LMCACHE_MAX_LOCAL_CPU_SIZE=4
LMCACHE_REMOTE_URL=fs://local:0/mnt/ws5000/kvpool_fs   # Both instances share the same pool
LMCACHE_REMOTE_SERDE=naive
(LMCACHE_LOCAL_DISK is no longer configured)
```

Everything else is identical to the baseline: TP4×2 dual instance, `--no-enable-prefix-caching`, `gpu-memory-utilization 0.9`, `max-model-len 32768`, `PYTHONHASHSEED=0`.

### 3.2 Data Preparation and Three Measurement Rounds

- Both instances prefilled 48 distinct sessions in parallel (A prefilled 0–23, B prefilled 24–47), shared pool measured on disk **344 GB / 22656 files**;
- Before each measurement round: `sync; echo 3 > /proc/sys/vm/drop_caches` to clear 1.5 TB page cache, ensuring physical cold reads;
- **Round 1 OWN 8+8**: Each reads its own prefilled sessions (A reads 0–7, B reads 24–31) — same scope as baseline;
- **Round 2 OWN 16+16**: Same as above, full load (A reads 0–15, B reads 24–39);
- **Round 3 CROSS 16+16 (hot sharing proof)**: **Cross cold read — A reads B's prefilled 24–39, B reads A's prefilled 0–15**, the two instances' read sets are disjoint and neither reads its own writes;
- Evidence collection: Per-request statistics of `hit tokens: 30xxx` (full hit) vs. `hit tokens: 0` (miss), parallel collection of `iostat -x 1 /dev/md0`.

---

## 4. Test Results

### 4.1 Summary Table for All Three Rounds (All Physical Cold Reads)

| Tier | Read Relationship | TTFT p50 A / B (s) | Aggregate Throughput | Disk Peak | Disk Busy Window Avg | Full Hit Verification |
|------|------|------|------|------|------|------|
| OWN 8+8 | Each reads its own | 13.73 / 13.59 | 67.4 tok/s | 10.37 GB/s | 9.21 GB/s | 16/16, zero misses |
| OWN 16+16 | Each reads its own | 26.70 / 25.03 | 71.6 tok/s | 10.36 GB/s | 8.98 GB/s | 32/32, zero misses |
| **CROSS 16+16** | **Cross-reads each other's** | **27.38 / 27.09** | **69.9 tok/s** | **10.41 GB/s** | **9.22 GB/s** | **32/32, zero misses** |

(Aggregate throughput = 2×N×64 ÷ batch completion time of the slower of the two instances)

### 4.2 Hot Sharing Verification (Core Results)

- **All 32 cross-requests achieved full disk hits** (hit tokens per request = 30208, hit rate 100%), with zero degraded recompute;
- Cross-read vs self-read: TTFT p50 27.2 s vs 25.9 s (mean basis), **sharing overhead ~5%** — source is only minor stat/scheduling differences, data path is identical;
- Array supply during cross-read did not drop but remained flat: peak 10.41 GB/s, busy window 9.22 GB/s, the highest among the three tiers — **under shared pool mode, array supply capability is unaffected by read relationship**;
- Compared to yesterday's `LocalDiskBackend` cross-instance attempt (hit=0, TTFT 67 s, degraded recompute): **same problem, closed by switching backends**.

### 4.3 Performance Comparison Under Same Metrics: fs:// Shared Pool vs LocalDiskBackend (Morning Baseline)

| Metric (Full-machine concurrent 16 tiers) | LocalDiskBackend | **fs:// Shared Pool** |
|------|------|------|
| TTFT p50 (A/B) | 13.17 / 13.89 s | 13.73 / 13.59 s |
| Aggregate Throughput | 66.5 tok/s | 67.4 tok/s |
| Disk Peak / Busy Window Avg | 10.33 / 8.55 GB/s | 10.37 / 9.21 GB/s |

| Metric (Full-machine concurrent 32 tiers) | LocalDiskBackend | **fs:// Shared Pool** |
|------|------|------|
| TTFT p50 (A/B) | 26.01 / 27.80 s | 26.70 / 25.03 s |
| Aggregate Throughput | 69.0 tok/s | 71.6 tok/s |
| Disk Peak / Busy Window Avg | 10.27 / 8.88 GB/s | 10.36 / 8.98 GB/s |

Across both tiers, TTFT, throughput, and array supply for the two backends are all within measurement noise (fs:// is even slightly better). **Conclusion: Cross-instance sharing capability is obtained with zero performance cost.** The asyncio+aiofiles read path of the fs connector and the thread pool read path of LocalDiskBackend both achieve 90%+ of the single-port line rate of the array under this workload.

### 4.4 Ingestion (Write Path)

Parallel ingestion of 48 sessions by two instances took 363 s, consistent with the baseline (361–363 s); the write path uses asynchronous put (temp file + atomic rename), and all 22656 files were fully written to disk with no corruption and no residual `.tmp` files.

---

## V. Value and Boundaries

### 5.1 Direct Significance for Production Architecture

1. **Pre-fill cost paid only once per machine**: Long contexts (long system prompts, RAG documents, historical sessions) processed by any instance are immediately hit by others — the KV reuse rate of a multi-instance cluster is elevated from "within-instance" to "whole-machine";
2. **Sessions can be freely migrated**: The next request of a session routed to any instance can recover a 30K context in 27 s (instead of 270 s for recompute) — load balancers no longer require session affinity constraints;
3. **Instance rolling restart no longer clears cache**: The pool resides on the array, and the index is the file system; after an instance restart, all historical KV remains visible (LocalDiskBackend loses everything upon restart);
4. **Leveraging AISSD5000 capacity and bandwidth**: A shared pool starting at 14 TB + single-port 10.4 GB/s (expandable to 6 ports) supply is an architectural capability that local disk solutions (2 TB per machine, 6.78 GB/s, no cross-instance semantic sharing) cannot provide.

### 5.2 Usage Boundaries (Engineering Constraints, All Controlled in Experiments)

1. **Same-machine multi-instance is plug-and-play; cross-node requires a shared file system layer** — AISSD5000 is a block device; mounting the same XFS from multiple hosts simultaneously will corrupt the file system; cross-node sharing requires an NFS gateway/cluster file system layer, or switching to a centralized index backend (Mooncake/Redis class), to be verified next;
2. **Sharing parties must have identical geometry**: The key format includes world_size/worker_id/chunk hash — same TP parallelism, same chunk size, same `PYTHONHASHSEED`, same model;
3. **Pool lifecycle requires external management**: The remote pool has no automatic LRU eviction (`MAX_LOCAL_DISK_SIZE` does not apply); periodic cleanup by directory/mtime is needed; a 14 TB pool can hold approximately 1900 30K sessions under this load, no pressure in the short term;
4. **Upstream URL validation bug**: Must write `fs://local:0/path` placeholder form (§2); recommended to file an upstream issue.

---

## VI. Conclusion

1. **Solution 1 is feasible and verified**: LMCache's built-in `fs://` remote connector + AISSD5000 shared pool, **zero code changes**, enables cross-instance KV hot sharing — 32/32 full hits for cross cold reads, closing the gap from §4.6 yesterday;
2. **Sharing has zero performance cost**: Compared to LocalDiskBackend under the same metrics, TTFT/throughput/array supply (10.4 GB/s peak) are all equal; cross-read is only +5% TTFT relative to self-read;
3. **Architectural value**: Pre-fill paid once per machine, free session migration, cache persistence across restarts — these capabilities are built on "shared medium + file system as index," which local disk solutions cannot semantically provide;
4. **Boundaries are clear**: Same-machine plug-and-play; cross-node requires a shared file system layer or centralized index, which is the next verification item.

---

## Appendix A: Reproduction Commands (Colleague's Server, Inside `vllm` Container)

### A.1 Dual Instance Startup (Shared Pool)

```bash
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8
mkdir -p /mnt/ws5000/kvpool_fs
for I in 0 1; do
  DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7"); PORT=$((8000+I))
  docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 \
LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
LMCACHE_REMOTE_URL=fs://local:0/mnt/ws5000/kvpool_fs LMCACHE_REMOTE_SERDE=naive; \
vllm serve $MODEL --served-model-name qwen \
 --tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
 --max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
 --port $PORT > /mnt/ws5000/fsws_i$I.log 2>&1"
  sleep 30
done
## Note: The URL cannot be written as fs:///path per the official documentation (that version's parse_remote_url enforces host:port validation and will fail).
## It must be written in the placeholder form fs://local:0/path.
```

### A.2 Populate and Cross Cold Read Measurement

```bash
## Parallel populate 48 sessions (A: 0-23, B: 24-47)
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 populate 530 24 0  > /mnt/ws5000/results/fs_ppA.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 populate 530 24 24 > /mnt/ws5000/results/fs_ppB.log 2>&1"
## Cross cold read (hot sharing proof): A reads B's populated sessions 24-39, B reads A's populated sessions 0-15
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
iostat -x 1 400 /dev/md0 > /tmp/io_fs_CROSS.log &
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 measure 530 16 24 64 16 > /mnt/ws5000/results/fs_CROSS_A.log 2>&1"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 16 0  64 16 > /mnt/ws5000/results/fs_CROSS_B.log 2>&1"
## Forensics (each instance's 16 requests should all be full hits):
grep -acE 'hit tokens: 30[0-9]{3}' /mnt/ws5000/fsws_i0.log
grep -acE 'hit tokens: 0,' /mnt/ws5000/fsws_i0.log   # Expected 0
```

### A.3 Unit Test Reproduction of the URL Validation Bug

```bash
docker exec vllm python3 -c "
from lmcache.v1.storage_backend.connector import parse_remote_url
parse_remote_url('fs:///mnt/ws5000/kvpool_fs')   # AssertionError: missing host (documentation example format)
parse_remote_url('fs://local:0/mnt/ws5000/kvpool_fs')  # OK, path='/mnt/ws5000/kvpool_fs'
"
```

## Appendix B: Raw Data Archive (Colleague Server)

| File | Content |
|------|---------|
| `/tmp/fsshare.out` | Full experiment orchestration log |
| `/mnt/ws5000/results/fs_{OWN_c8x2,OWN_c16x2,CROSS_c16x2}_{A,B}.log` | Raw client output for each configuration (two instances) |
| `/mnt/ws5000/fsws_i{0,1}.log` | Full service logs for both instances (including per-request hit forensics lines and FS connector initialization lines) |
| `/tmp/io_fs_*.log` | Per-second iostat raw records for each configuration |
| `/mnt/ws5000/kvpool_fs/` | Shared pool (344 GB / 22656 files, retained for review after the experiment) |
