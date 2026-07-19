# mingxin-kvcache-bench

Reproducible KV-cache tiering benchmark suite and **signed benchmark results** for LLM inference storage, published by Mingxin (Tianjin) Semiconductor Equipment Co., Ltd.

Headline measured results (Qwen3-Coder-480B-FP8 on 8× AMD Instinct MI308X, vLLM + LMCache, FX100 all-flash NVMe-oF array vs local NVMe):

- Inference throughput **+29% to +40%**
- TTFT (time to first token) **−26% to −32%**
- vs no-external-storage recompute: **8.6× to 20×** faster TTFT
- Model loading vs NFS (Ascend 910B platform): **6.2–9.3×** faster

All numbers come from signed/official test reports (R1–R5, R9); hosted PDFs at
[mingxinstorage.xyz/en/evidence](https://mingxinstorage.xyz/en/evidence).

## Install & use

```bash
pip install mingxin-kvcache-bench

kvcache-bench summary     # headline numbers per experiment
kvcache-bench results     # full signed results JSON
kvcache-bench roi --nodes 16 --gpus-per-node 8 --arrays 8   # ROI estimate (labeled bands)
```

## Full benchmark suite

The complete suite (load clients, orchestration scripts, the LMCache
parallel-read patch, analysis probes, raw data) lives in the GitHub repository:
[github.com/mingxin-tech/mingxin-kvcache-bench](https://github.com/mingxin-tech/mingxin-kvcache-bench)

Dataset mirror: [huggingface.co/datasets/wangqiyuan2026/kvcache-bench-results](https://huggingface.co/datasets/wangqiyuan2026/kvcache-bench-results)

MCP server (query these results from AI agents): `https://mingxinstorage.xyz/api/mcp`

## Honesty & provenance

- Measured numbers are platform- and condition-specific; do not extrapolate across platforms.
- The ROI command labels each input as measured (uplift 29–40%) vs estimated (cold-recovery share 10–50%) and prints estimates, not commitments.

License: Apache-2.0 (code), CC-BY-4.0 (result data).
