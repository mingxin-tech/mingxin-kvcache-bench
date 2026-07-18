#!/bin/bash
# 只读：找八卡单实例(480B) ws vs local 实验的日志与数据
echo "===l480 系列日志==="
ls -lt /mnt/ws5000/l480*.log /mnt/ws5000/*480*.log 2>/dev/null | head -10
echo
echo "===ws 版日志里的 LMCache 配置与 retrieve==="
for L in $(ls -t /mnt/ws5000/l480*ws*.log /mnt/ws5000/l480*.log 2>/dev/null | head -3); do
  echo "--- $L ---"
  grep -aoE 'LMCACHE_[A-Z_]+=[^ ]+' "$L" 2>/dev/null | head -8
  grep -aoE 'lmcache.*local_disk|LOCAL_DISK|local_cpu[^,]*|chunk_size[^,]*' "$L" 2>/dev/null | head -4
  echo "retrieve 样例:"
  grep -aoE 'Retrieved [0-9]+ out of [0-9]+ required tokens[^;]*throughput: [0-9.]+ GB/s' "$L" 2>/dev/null | tail -5
  echo "错误:"
  grep -aicE 'error|failed' "$L" 2>/dev/null
done
echo
echo "===启动命令历史（bash history 里的 l480/lmcache 相关）==="
grep -aE '480|LMCACHE' /root/.bash_history 2>/dev/null | tail -20 || sudo -n grep -aE '480|LMCACHE' /root/.bash_history 2>/dev/null | tail -20
echo
echo "===sla 目录==="
ls /mnt/ws5000/lmcache_sla 2>/dev/null | head -5
du -sh /mnt/ws5000/lmcache_sla 2>/dev/null
echo
echo "===results 里 480 相关==="
ls -lt /mnt/ws5000/results/ 2>/dev/null | head -15
echo "===DONE==="
