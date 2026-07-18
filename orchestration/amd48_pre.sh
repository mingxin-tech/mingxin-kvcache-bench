#!/bin/bash
echo "===当前 vllm 进程==="
ps aux | grep 'vllm serve' | grep -v grep | head -4 | cut -c1-350
echo "===GPU 利用率与显存==="
rocm-smi 2>/dev/null | grep -E '^[0-7] ' | awk '{v=$(NF-1); u=$NF; print "GPU"$1, "VRAM", v, "UTIL", u}'
echo "===最近10分钟是否有请求（最新日志尾）==="
L=$(ls -t /mnt/ws5000/*.log 2>/dev/null | head -1); echo "log=$L"
ls -l --time-style=+%H:%M:%S "$L" 2>/dev/null
tail -c 1200 "$L" 2>/dev/null | tr '\r' '\n' | grep -aE 'throughput|Running' | tail -3
