# mingxin-kvcache-bench

Reproducible benchmark suite and measured results for **LLM KV Cache tiered-storage acceleration** on AMD Instinct MI308X, by [Mingxin Technology](https://mingxinstorage.xyz) (铭信).

大模型推理 **KV Cache 分层存储加速**的可复现基准套件与实测结果（AMD Instinct MI308X ×8 / ROCm 7.2），由[铭信](https://mingxinstorage.xyz)发布。

## What's inside / 内容

| Directory | Contents |
|---|---|
| `lmcache_patch/` | **LMCache parallel-read patch** for `local_disk_backend.py` (+53 −2, `git apply`-able). Measured effect: cold-read TTFT 37.97 s → 9.30 s (**4.1×**), disk read bandwidth 0.98 → 5.23 GB/s (**5.3×**) at concurrency 16, Qwen2.5-32B, single GPU. Includes ORIGINAL/PATCHED full files and unified diff. |
| `load_clients/` | Load-generation clients: populate/measure phases, per-request TTFT/E2E/TPOT capture, multi-instance port-parameterized runners, SLA-mode client, prefix-hit scanner. |
| `orchestration/` | End-to-end experiment scripts for every run in the reports (TP8 / TP4×2 / fs:// shared pool / cleanup & restore). |
| `analysis_probe/` | Metric aggregation (whole-machine p50/p90/p99, TPOT), cold-read forensics (per-request full-hit counting), LMCache backend/index probes. |
| `results/` | **Structured measured results** (JSON + CSV) extracted from the signed/official reports — usable directly as a dataset. |
| `reports/` | Full test reports (Markdown + PDF, Chinese): 480B TP8 long-context, TP4×2 dual-instance, full-metric summary, fs:// cross-instance hot-sharing verification. |
| `docs_EXPORT_README.md` | Original export manifest with file-by-file descriptions and reproduction walkthrough. |

## Headline results / 核心实测结果

Platform: 8× AMD Instinct MI308X (192 GB HBM, gfx942), ROCm 7.2, vLLM 0.20.1+rocm721, LMCache (upstream mainline, 2026-06-29 source build). Model: Qwen3-Coder-480B-FP8 (~450 GB weights). Device under test: Mingxin FX100 all-flash NVMe-oF array (4-disk RAID0, RoCEv2, 100 GbE) vs local NVMe (PCIe Gen4) vs recompute-only.

480B, production deployment form (TP8), long-context cold recovery:

| Concurrency | FX100 TTFT p50 | Local NVMe TTFT p50 | Recompute TTFT p50 | FX100 tput | Local tput |
|---|---|---|---|---|---|
| 8 | 7.53 s | 10.17 s | — | 56.6 tok/s | 43.9 tok/s |
| 16 | 11.85 s | 17.31 s | 149.48 s | 74.9 tok/s | 53.6 tok/s |
| 32 | 26.35 s | 35.73 s | — | 71.6 tok/s | 53.9 tok/s |

- Throughput **+29% to +40%** and TTFT **−26% to −32%** vs local NVMe, consistent across TP8 and TP4×2 forms. Mechanism: local disk pins at 6.78 GB/s; the array sustains 10.2–10.4 GB/s.
- **8.6×–20×** faster TTFT vs recompute-without-external-KV.
- `fs://` shared pool achieves cross-instance KV hot-sharing with zero performance penalty (32/32 cross-read full hits); two independent measurement rounds deviate ≤ 5%.

All numbers are measured on the stated platform and conditions — they are not general claims. See `reports/` for full methodology and `results/` for machine-readable data.

## Reproduce / 复现

```bash
# 1) Apply the LMCache parallel-read patch (inside your LMCache checkout)
git apply lmcache_patch/local_disk_backend.git.patch

# 2) Start instances and run a full experiment (edit paths/IPs for your environment;
#    scripts expect SUDO_PW env var for the sudo askpass helper)
export SUDO_PW=<your-sudo-password>
bash orchestration/amd48_full2.sh

# 3) Aggregate whole-machine percentiles and verify cold reads
bash analysis_probe/agg_metrics.sh
bash analysis_probe/verify_coldread.sh
```

The orchestration scripts contain environment-specific paths (container name `vllm`, mount points `/mnt/ws5000`, `/srv2`) that must be adapted; `docs_EXPORT_README.md` documents each script's role.

## LMCache patch upstreaming / 补丁回馈

The parallel-read patch targets the LMCache mainline as of 2026-06-29 (`local_disk_backend.py`). It is published here as an independent, verifiable artifact first; upstream contribution will follow after rebasing and re-validating against LMCache HEAD. Anyone is welcome to pick it up under Apache-2.0.

## License / 许可

- Code and scripts: **Apache-2.0** (see `LICENSE`).
- `results/` data: **CC-BY-4.0** — cite "Mingxin Technology, mingxin-kvcache-bench" and link this repository.
- `reports/` PDFs/Markdown are © Mingxin Technology, redistributed here for verification purposes.

## Links

- Hugging Face dataset: https://huggingface.co/datasets/wangqiyuan2026/kvcache-bench-results
- Website: https://mingxinstorage.xyz (evidence library: https://mingxinstorage.xyz/evidence)
- ROI calculator: https://mingxinstorage.xyz/roi
- Contact: see https://mingxinstorage.xyz/contact
