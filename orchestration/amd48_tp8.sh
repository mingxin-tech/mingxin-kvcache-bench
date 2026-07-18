#!/bin/bash
# 只读检查：八卡单实例实验现状
S="echo ${SUDO_PW} | sudo -S"
echo "===正在运行的 vllm 进程与参数==="
ps aux | grep -E 'vllm serve' | grep -v grep | head -3 | cut -c1-400
echo
echo "===GPU 占用==="
rocm-smi 2>/dev/null | grep -E '^[0-7] ' | awk '{print $1, "VRAM%", $(NF-1), "GPU%", $NF}'
echo
echo "===最近的 vllm 日志==="
ls -lt /mnt/ws5000/vllm*.log 2>/dev/null | head -5
echo
echo "===最新日志尾部（retrieve/need/错误）==="
L=$(ls -t /mnt/ws5000/vllm*.log 2>/dev/null | head -1)
echo "log=$L"
tail -c 4000 "$L" 2>/dev/null | tr '\r' '\n' | grep -aE 'Retrieved|need to load|error|Error|throughput' | tail -12
echo
echo "===LMCache 相关环境（从进程 environ 读）==="
P=$(pgrep -f 'vllm serve' | head -1)
[ -n "$P" ] && cat /proc/$P/environ 2>/dev/null | tr '\0' '\n' | grep -E 'LMCACHE|HIP_VISIBLE' | head -15
echo
echo "===EngineCore 数量（TP rank 数）==="
pgrep -f EngineCore | wc -l
echo
echo "===lmcache 磁盘目录==="
ls -d /mnt/ws5000/lmcache* 2>/dev/null; du -sh /mnt/ws5000/lmcache 2>/dev/null | head -2
find /mnt/ws5000/lmcache -type f 2>/dev/null | head -3
find /mnt/ws5000/lmcache -type f 2>/dev/null | wc -l
echo "单chunk文件大小样例:"
find /mnt/ws5000/lmcache -type f 2>/dev/null | head -1 | xargs stat -c %s 2>/dev/null
echo
echo "===md0 当前读带宽采样（5秒）==="
iostat -x 1 5 /dev/md0 2>/dev/null | awk '$1=="md0"{printf "%.2f GB/s util=%s%%\n", $3/1e6, $NF}'
echo
echo "===最近的测试输出==="
ls -lt /mnt/ws5000/results/ 2>/dev/null | head -6
echo "===DONE==="
