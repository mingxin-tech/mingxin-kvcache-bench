#!/bin/bash
# 只读：解析 480B ws vs local 四份日志的关键内容
for L in /mnt/ws5000/load480_ws.log /mnt/ws5000/load480_local.log /mnt/ws5000/l480v2_ws.log /mnt/ws5000/l480v2_local.log; do
  echo "=============== $L ==============="
  echo "--- 模型路径与关键参数 ---"
  grep -aoE 'vllm serve [^ ]+|model=[^,]+|--tensor-parallel-size [0-9]+' "$L" 2>/dev/null | head -3
  grep -aoE 'non_default_args[^}]{0,300}' "$L" 2>/dev/null | head -1 | cut -c1-300
  echo "--- 权重加载耗时 ---"
  grep -aE 'Loading weights took|Model loading took|load_model|Loading safetensors|Time spent|took .* seconds|startup complete' "$L" 2>/dev/null | tr '\r' '\n' | grep -aE 'took|complete' | tail -8
  echo "--- LMCache 痕迹 ---"
  grep -aicE 'lmcache' "$L" 2>/dev/null
  echo "--- 错误样例 ---"
  grep -aiE 'error|failed' "$L" 2>/dev/null | tr '\r' '\n' | grep -aiE 'error|failed' | head -3 | cut -c1-200
done
echo "===模型文件位置对比==="
ls -d /mnt/ws5000/models/*480* /srv2/*480* 2>/dev/null
du -sh /srv2/Qwen3-Coder-480B-FP8 2>/dev/null
echo "===DONE==="
