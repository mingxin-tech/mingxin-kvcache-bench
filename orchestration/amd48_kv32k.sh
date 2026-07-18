#!/bin/bash
# 实验一+二：480B TP8 · 32K 长上下文冷恢复 + 并发梯度（WS / 本地 / 重算对照）
# reps=530 ≈ 29.8K tok/会话, KV≈7.1GB；populate 34；conc 8/16/32 各轮前 drop_caches
say(){ echo "[$(date +%H:%M:%S)] $*"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"

# 同事今早的服务命令（结束后恢复）
cat > /tmp/colleague_restore.sh <<'RESTORE_EOF'
#!/bin/bash
export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4
export LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache_kv32 LMCACHE_MAX_LOCAL_DISK_SIZE=800
vllm serve /mnt/ws5000/models/Qwen2.5-32B-Instruct --served-model-name qwen \
  --tensor-parallel-size 8 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.9 --port 8000 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  > /mnt/ws5000/kv32_ws.log 2>&1
RESTORE_EOF
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw2 2>/dev/null
sudo -A cp /tmp/colleague_restore.sh /mnt/ws5000/_restore_colleague.sh
sudo -A chmod +x /mnt/ws5000/_restore_colleague.sh
say "同事服务命令已留档到 /mnt/ws5000/_restore_colleague.sh"

kill_all(){
  for PAT in 'vllm serve' 'VLLM::EngineCore' 'Worker_TP' 'from multiprocessing'; do
    $S docker exec vllm bash -c "pkill -9 -f '$PAT' 2>/dev/null" 2>/dev/null
    $S pkill -9 -f "$PAT" 2>/dev/null
  done
  sleep 12
}

wait_ready(){ for i in $(seq 1 160); do curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen && { say READY; return 0; }; sleep 5; done; say TIMEOUT; return 1; }

measure(){ # $1=标签 $2=N $3=conc $4=iostat设备 $5=日志
  TAG=$1; N=$2; C=$3; DEV=$4; LOG=$5
  MARK=$($S bash -c "wc -l < $LOG")
  sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
  nohup bash -c "iostat -x 1 240 $DEV > /tmp/io32k_$TAG.log 2>&1" >/dev/null 2>&1 &
  IOP=$!
  T0=$(date +%s.%N)
  $S docker exec vllm python3 /mnt/ws5000/benchcap_off.py $TAG 530 $N 64 $C 0
  T1=$(date +%s.%N)
  sleep 2; kill $IOP 2>/dev/null
  D=$(basename $DEV)
  W=$(awk "BEGIN{printf \"%.1f\", $T1-$T0}")
  TPS=$(awk "BEGIN{printf \"%.1f\", $N*64/($T1-$T0)}")
  say "$TAG: wall=${W}s 输出吞吐=${TPS} tok/s"
  say "$TAG 盘峰值: $(awk -v X=$D '$1==X{if($3>m)m=$3}END{printf "%.2f GB/s", m/1e6}' /tmp/io32k_$TAG.log)  忙窗均值: $(awk -v X=$D '$1==X && $3>500000{c++;s+=$3}END{if(c)printf "%.2f GB/s (%ds)", s/c/1e6, c}' /tmp/io32k_$TAG.log)"
  say "$TAG need_gt0=$($S bash -c "tail -n +$MARK $LOG | grep -acE 'need to load: [1-9]'")"
}

run_disk(){ # $1=轮名 $2=盘URI $3=iostat设备 $4=日志
  RN=$1; DISK=$2; DEV=$3; LOG=$4
  say "########## $RN ##########"
  kill_all
  $S bash -c "rm -rf ${DISK#file://}; mkdir -p ${DISK#file://}; chmod 777 ${DISK#file://}"
  $S docker exec -d vllm bash -c "export VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO \
LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
LMCACHE_LOCAL_DISK=$DISK LMCACHE_MAX_LOCAL_DISK_SIZE=1000; \
vllm serve /srv2/Qwen3-Coder-480B-FP8 --served-model-name qwen \
--tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
> $LOG 2>&1"
  wait_ready || return 1
  say "populate 34 会话（reps=530≈29.8K tok）"
  T0=$(date +%s)
  $S docker exec vllm python3 /mnt/ws5000/benchcap_full.py ${RN}_pp 530 34 populate 1 1
  say "populate 耗时 $(( $(date +%s)-T0 ))s 落盘=$($S du -sh ${DISK#file://} | cut -f1)"
  measure ${RN}_c8  8  8  $DEV $LOG
  measure ${RN}_c16 16 16 $DEV $LOG
  measure ${RN}_c32 32 32 $DEV $LOG
}

# ===== WS 轮 =====
run_disk WS32K file:///mnt/ws5000/lmcache480 /dev/md0 /mnt/ws5000/kv32k_ws.log
# ===== 本地轮 =====
df -h /srv2 | tail -1
run_disk LOC32K file:///srv2/lmcache480_local /dev/nvme1n1 /mnt/ws5000/kv32k_loc.log

# ===== 重算对照（conc16 一档）=====
say "########## REC32K ##########"
kill_all
$S docker exec -d vllm bash -c "export VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO \
LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4; \
vllm serve /srv2/Qwen3-Coder-480B-FP8 --served-model-name qwen \
--tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
> /mnt/ws5000/kv32k_rec.log 2>&1"
wait_ready && measure REC32K_c16 16 16 /dev/md0 /mnt/ws5000/kv32k_rec.log

# ===== 恢复同事的 32B 服务 =====
say "########## 恢复同事服务 ##########"
kill_all
$S docker exec -d vllm bash /mnt/ws5000/_restore_colleague.sh
for i in $(seq 1 80); do curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen && { say COLLEAGUE_RESTORED; break; }; sleep 5; done
rm -f /tmp/.pw /tmp/.ap
say "ALL_DONE_32K"
