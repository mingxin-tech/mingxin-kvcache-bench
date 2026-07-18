#!/bin/bash
# AMD 服务器单卡单实例复现（严格按报告 §6.0/§6.1；新目录/新日志，不动原有文件）
say(){ echo "[$(date +%H:%M:%S)] $*"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"

say "STEP0 重启容器清理 8 个僵尸 EngineCore（手册标准处置，不动文件）"
$S docker restart vllm >/dev/null; sleep 12
$S docker exec vllm bash -c "rocm-smi --showmeminfo vram 2>/dev/null | grep 'GPU\[0\]' | tail -1"

say "STEP1 校验补丁在位（只读）"
$S docker exec vllm grep -c 'PAR-READ PATCH' /root/LMCache/lmcache/v1/storage_backend/local_disk_backend.py

say "STEP2 起服（报告 §6.0 原样：GPU0 32B util0.9 CPU4 disk500 关prefix）"
$S docker exec vllm bash -c "mkdir -p /mnt/ws5000/lmcache_repro"
$S docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 \
 LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
 LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache_repro LMCACHE_MAX_LOCAL_DISK_SIZE=500; \
 vllm serve /mnt/ws5000/models/Qwen2.5-32B-Instruct --served-model-name qwen \
 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
 --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' --port 8000 \
 > /mnt/ws5000/vllm_repro48.log 2>&1"
for i in $(seq 1 120); do
  $S docker exec vllm bash -c "grep -qa 'Application startup complete' /mnt/ws5000/vllm_repro48.log" 2>/dev/null && { say READY; break; }
  sleep 5
  [ $i -eq 120 ] && { say TIMEOUT; $S docker exec vllm bash -c "grep -aiE 'error|Traceback' /mnt/ws5000/vllm_repro48.log | tail -6"; exit 1; }
done

say "STEP3 populate 90（reps=150 ≈8448 tok，KV≈2.06GB/会话，共 ~186GB）"
$S docker exec vllm python3 /mnt/ws5000/benchcap_full.py POPR 150 90 populate 1 1
say "落盘=$($S docker exec vllm bash -c 'du -sh /mnt/ws5000/lmcache_repro | cut -f1')"

say "STEP4 drop_caches + iostat + 冷读 sess0-15 conc16（报告 §6.1 原样）"
MARK=$($S docker exec vllm bash -c "wc -l < /mnt/ws5000/vllm_repro48.log")
sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
free -g | sed -n 2p
nohup bash -c "iostat -x 1 90 /dev/md0 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 > /tmp/io_repro16.log 2>&1" >/dev/null 2>&1 &
$S docker exec vllm python3 /mnt/ws5000/benchcap_off.py PAR_R16 150 16 64 16 0
sleep 2
say "md0 峰值: $(awk '$1==\"md0\"{if($3>m)m=$3}END{printf \"%.2f GB/s\", m/1e6}' /tmp/io_repro16.log)"
say "md0 忙窗均值: $(awk '$1==\"md0\" && $3>200000{s+=$3;c++}END{if(c)printf \"%.2f GB/s (n=%d)\", s/c/1e6, c}' /tmp/io_repro16.log)"
say "成员盘峰值:"; for d in nvme2n1 nvme3n1 nvme4n1 nvme5n1; do awk -v D=$d '$1==D{if($3>m)m=$3}END{printf "  %s %.0f MB/s\n", D, m/1e3}' /tmp/io_repro16.log; done
say "取证: need_gt0=$($S docker exec vllm bash -c "tail -n +$MARK /mnt/ws5000/vllm_repro48.log | grep -acE 'need to load: [1-9]'")  alloc_fail=$($S docker exec vllm bash -c "tail -n +$MARK /mnt/ws5000/vllm_repro48.log | grep -ac 'Failed to allocate memory block'")"
say "retrieve样例:"; $S docker exec vllm bash -c "tail -n +$MARK /mnt/ws5000/vllm_repro48.log | grep -aoE 'throughput: [0-9.]+ GB/s' | tail -6"
say "DONE_AMD48_REPRO"
