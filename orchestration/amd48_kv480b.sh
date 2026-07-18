#!/bin/bash
# 480B TP8 · KV回读 WS vs 本地 A/B · v2：容器内跑（含LMCache+补丁），宿主恢复原服务
say(){ echo "[$(date +%H:%M:%S)] $*"; }
export SUDO_ASKPASS=/tmp/.ap
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
S="sudo -A"

ORIG_CMD="VLLM_ROCM_USE_AITER=1 /usr/local/bin/vllm serve /srv2/Qwen3-Coder-480B-FP8 --served-model-name q3c --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code --max-model-len 32768 --gpu-memory-utilization 0.9"

say "确认容器内环境（vllm/LMCache/模型可见性）"
$S docker exec vllm bash -c "which vllm; python3 -c 'import lmcache; print(lmcache.__file__)'; ls /srv2/Qwen3-Coder-480B-FP8/config.json" || { say CONTAINER_ENV_FAIL; exit 1; }

kill_all(){
  for PAT in 'vllm serve' 'VLLM::EngineCore' 'Worker_TP' 'from multiprocessing'; do
    $S docker exec vllm bash -c "pkill -9 -f '$PAT' 2>/dev/null" 2>/dev/null
    $S pkill -9 -f "$PAT" 2>/dev/null
  done
  sleep 12
  V=$(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}')
  say "清理后 GPU0 VRAM: $V"
}

wait_ready(){ L=$1; for i in $(seq 1 160); do curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen && { say READY; return 0; }; sleep 5; done; say TIMEOUT; $S docker exec vllm bash -c "tail -c 2000 $L | tr '\r' '\n' | grep -aiE 'error|Traceback' | tail -5"; return 1; }

run_round(){ # $1=标签 $2=盘URI $3=iostat设备 $4=日志(容器路径=宿主路径,均在/mnt/ws5000)
  TAG=$1; DISK=$2; DEV=$3; LOG=$4
  say "===== $TAG ====="
  kill_all
  $S bash -c "rm -rf ${DISK#file://}; mkdir -p ${DISK#file://}; chmod 777 ${DISK#file://}"
  $S docker exec -d vllm bash -c "export VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO \
LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
LMCACHE_LOCAL_DISK=$DISK LMCACHE_MAX_LOCAL_DISK_SIZE=500; \
vllm serve /srv2/Qwen3-Coder-480B-FP8 --served-model-name qwen \
--tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
> $LOG 2>&1"
  wait_ready $LOG || return 1
  say "populate 24（8448 tok/会话）"
  $S docker exec vllm python3 /mnt/ws5000/benchcap_full.py ${TAG}_pp 150 24 populate 1 1
  say "落盘=$($S du -sh ${DISK#file://} | cut -f1)"
  MARK=$($S bash -c "wc -l < $LOG")
  sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
  nohup bash -c "iostat -x 1 150 $DEV > /tmp/io_kv480_$TAG.log 2>&1" >/dev/null 2>&1 &
  say "冷读 16 会话 conc16"
  $S docker exec vllm python3 /mnt/ws5000/benchcap_off.py $TAG 150 16 64 16 0
  sleep 2
  D=$(basename $DEV)
  say "$TAG 盘峰值: $(awk -v X=$D '$1==X{if($3>m)m=$3}END{printf "%.2f GB/s", m/1e6}' /tmp/io_kv480_$TAG.log)"
  say "$TAG need_gt0=$($S bash -c "tail -n +$MARK $LOG | grep -acE 'need to load: [1-9]'")"
  say "$TAG retrieve样例:"
  $S bash -c "tail -n +$MARK $LOG | grep -aoE 'Retrieved [0-9]+ out of [0-9]+[^;]*throughput: [0-9.]+ GB/s' | tail -4"
}

run_round WS480 file:///mnt/ws5000/lmcache480 /dev/md0 /mnt/ws5000/kv480_ws.log
run_round LOC480 file:///srv2/lmcache480_local /dev/nvme1n1 /mnt/ws5000/kv480_loc.log

say "===== 恢复原 480B 服务（宿主）====="
kill_all
$S bash -c "nohup bash -lc '$ORIG_CMD > /mnt/ws5000/l480v2_local.log 2>&1' >/dev/null 2>&1 &"
for i in $(seq 1 80); do curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q q3c && { say RESTORED_OK; break; }; sleep 5; done
rm -f /tmp/.pw /tmp/.ap
say "ALL_DONE_KV480"
