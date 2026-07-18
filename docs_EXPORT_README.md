# AISSD5000 KV Cache 测试 — AMD 服务器代码与脚本导出包

本包汇总在 **AMD Instinct MI308X ×8 / ROCm 7.2 服务器（<AMD-SERVER-IP>）** 上完成的 **LLM KV Cache 分层存储基准测试**的全部代码/脚本、LMCache 源码补丁与测试报告。

- 推理栈：vLLM 0.20.1+rocm721 + LMCache（上游主线源码编译），容器 `vllm`
- 被测存储：AISSD5000 全闪 NVMe-oF 阵列（4 盘 RAID0，`/dev/md0`，挂 `/mnt/ws5000`）；对照本地 NVMe（`/srv2`）
- 模型：Qwen3-Coder-480B-FP8（TP8 单实例 / TP4×2 双实例）
- 导出时间：2026-07-07
- 范围：**仅 AMD 服务器**（MetaX 平台的 C 引擎/GDS 等工作不在本包）

---

## 0. 说明

- 唯一**修改的既有第三方源码**是 LMCache 的 `local_disk_backend.py`（并行读补丁，见 `lmcache_patch/`），其余均为本项目**新编写**的负载客户端/编排/分析脚本。
- 本包**不含模型权重与实验数据**（KV 池、iostat 日志等留在服务器）。
- 脚本设计为在服务器上执行（多经 `docker exec vllm` 或操作宿主）；内含针对该服务器的硬编码路径与 sudo 免密辅助（写 `/tmp/.pw`、`/tmp/.ap`），移植需调整。

---

## 1. 目录结构

```
kvcache_amd_exports/
├── README.md                       本文档
├── lmcache_patch/                  【核心源码改动】LMCache 并行读补丁
│   ├── local_disk_backend.ORIGINAL.py   原始(673 行)
│   ├── local_disk_backend.PATCHED.py    补丁后(722 行, 服务器现行)
│   ├── local_disk_backend.patch         unified diff
│   └── local_disk_backend.git.patch     git diff(权威, +53 −2, 可 git apply)
├── load_clients/                   负载客户端(灌入/测量, 源自 /mnt/ws5000)
│   ├── bench_mp.py                  多实例负载客户端(端口参数化; TTFT p50/p90/p99+聚合吞吐)
│   ├── bench_mp2.py                 增强版(逐请求 TTFT/E2E/TPOT, 全机分位合并)
│   ├── benchcap.py / _full / _off / _mp   单实例 populate/measure 系列
│   ├── bench_sla.py                SLA 口径测试
│   ├── prefixscan.py               前缀命中/长度扫描
│   ├── kvbench1.py / kvstat.py     KV 基准与统计采集
│   ├── serve.sh                    vLLM+LMCache 服务启动封装
│   └── launch_sla.sh / _np.sh      SLA 启动(np=no-prefix-caching)
├── orchestration/                  实验编排脚本(本轮全部实验)
│   ├── amd48_tp8.sh / _tp8b / _tp8c     480B TP8 单实例三方对照(WS/本地/重算)
│   ├── amd48_kv32k / _kv480 / _kv480b   480B 长上下文冷恢复各版本
│   ├── amd48_tp4x2.sh / _b / _c         480B TP4×2 双实例(WS/本地/重算)
│   ├── amd48_full2.sh                   全指标重测(四组×两档, 逐请求 TTFT/TPOT)
│   ├── amd48_fsshare.sh                 fs:// 共享池跨实例热共享验证
│   ├── amd48_fsrep / _64 / _64b         fs 共享池 TPOT 发散复测 + CPU 64G 修复验证
│   ├── amd48_lmc / _pre / _run / _recon 早期 LMCache 探查与运行
│   ├── amd48_clean.sh                   彻底清理残留 GPU 进程
│   ├── amd48_restore.sh / _final        恢复同事原服务 / 收尾
│   └── amd48_parse.sh                   结果解析
├── analysis_probe/                 分析/探查/取证
│   ├── agg_metrics.sh               合并 A/B 逐请求样本算全机分位(p50/p90/p99/TPOT)
│   ├── verify_coldread.sh           冷读取证(逐请求满额命中计数)
│   ├── probe_lmcache_share.sh/2     探 LMCache 后端索引机制与共享后端
│   ├── probe_fs_connector.sh        探 fs:// connector 配置接线/注册/尾块元数据
│   ├── check_fs_pool.sh / _2        核查 fs 共享池落盘
│   ├── check_fs_url.sh / _parse     诊断 fs:// URL 校验(missing host bug)
│   └── check_cpu64_alerts.sh        核查 CPU 中转层告警(分配失败/超时)
└── reports/                        AMD 服务器 KV Cache 测试报告(md + pdf)
    ├── ...（480B·TP8长上下文·正式版）
    ├── ...（480B·TP4x2双实例·正式版）
    ├── ...（480B·TP4x2·全指标汇总·正式版）
    ├── ...（480B·多实例形态·正式版）        ← 汇总 TP8 与 TP4x2 两形态
    └── ...（跨实例热共享验证·fs共享池·正式版）
```

---

## 2. 【核心源码改动】LMCache 并行读补丁

- **位置**：`vllm` 容器 `/root/LMCache/lmcache/v1/storage_backend/local_disk_backend.py`
- **改动**：新增覆写 `LocalDiskBackend.batched_get_blocking(keys)`（纯新增 49 行）——先串行分配各 chunk 的 CPU 缓冲，再用 32 线程池（`lmc_par_read`）并发读盘，榨取 4 盘 RAID0 聚合带宽。
- **应用**：`docker cp local_disk_backend.PATCHED.py vllm:/root/LMCache/lmcache/v1/storage_backend/local_disk_backend.py` 后重启服务；或 `cd /root/LMCache && git apply local_disk_backend.git.patch`。**回退**换回 `ORIGINAL.py` 或 `git checkout` 即可。
- **完整性核对（git 权威）**：AMD `vllm` 容器的 LMCache 为 git 仓库，`git status` 确认**全仓库仅此一个文件被修改**（`git diff` = +53 −2，即 `local_disk_backend.git.patch`）。容器内另有 `cache_engine.py.bak_A*`、`gpu_connectors.py.bak_A*` 两个备份文件，经逐一 diff 与当前文件**完全一致（diff=0）**——即被备份但内容从未实际改动，故不计入源码修改。comfyui 容器无任何源码修改。

---

## 3. 复现主流程（示例：TP4×2 全指标）

```bash
# 1) 启动双实例(独立池/fs共享/本地/重算, 见 orchestration/amd48_full2.sh 内 ENVS 切换)
# 2) 并行灌入 48 会话
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 populate 530 24 0"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 populate 530 24 24"
# 3) 每档测量(清页缓存 + iostat + 双实例并发)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8000 measure 530 16 0 64 16"
docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8001 measure 530 16 24 64 16"
# 4) 全机分位聚合 + 冷读取证
bash analysis_probe/agg_metrics.sh
bash analysis_probe/verify_coldread.sh
```

整套实验直接跑对应的 `orchestration/amd48_*.sh` 即可（各脚本内含启动/灌入/测量/取证/恢复全流程）。

---

## 4. 覆盖的实验与对应报告

| 实验 | 编排脚本 | 报告 |
|------|---------|------|
| 480B TP8 单实例·长上下文冷恢复(三方×并发梯度) | amd48_tp8*.sh / amd48_kv*.sh | 480B·TP8长上下文·正式版 |
| 480B TP4×2 双实例(三方) | amd48_tp4x2*.sh | 480B·TP4x2双实例·正式版 |
| TP4×2 全指标重测(四组×两档, 含TPOT) | amd48_full2.sh | 480B·TP4x2·全指标汇总·正式版 |
| 两形态汇总 | — | 480B·多实例形态·正式版 |
| fs:// 共享池跨实例热共享 + 发散复测/修复 | amd48_fsshare/fsrep*.sh | 跨实例热共享验证·fs共享池·正式版 |

---

## 5. 关键结论（详见 reports/）

- 480B 长上下文冷恢复下，AISSD5000 相比本地 NVMe **TTFT 降 26%–32%、吞吐升 29%–40%**，跨 TP8/TP4×2 两形态一致；机理为本地盘钉死 6.78 GB/s、AISSD5000 供到 10.2–10.4 GB/s。
- 相比无外存重算：TTFT 快 8.6–20 倍、吞吐高 17–21 倍、TPOT 好 75 倍以上。
- fs:// 共享池实现跨实例 KV 热共享且零性能代价（交叉读 32/32 全命中）；需注意 URL 必须写 `fs://local:0/path`（该版本校验 bug），且 CPU 中转层需 ≥ 并发在途 KV 体量（否则低并发 TPOT 发散，已定位并修复）。
