#!/bin/bash
# TP4×2 双实例实验：480B FP8 · 32K 冷恢复 · WS vs 本地 · 跨实例共享演示 · 与 TP8 基线同口径
say(){ echo "[$(date +%H:%M:%S)] $*"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8

# ---- Phase 0: 留档当前运行的服务（若有）----
PID=$(pgrep -f 'bash -lc.*vllm serve' | head -1)
if [ -n "$PID" ]; then
  $S bash -c "python3 -c \"
import sys
args=open('/proc/$PID/cmdline','rb').read().split(b'\0')
print(args[2].decode())\" > /mnt/ws5000/_restore_current.sh" 2>/dev/null
  say "已留档当前服务命令: $(head -c 120 /mnt/ws5000/_restore_current.sh 2>/dev/null)..."
  RESTORE=1
else
  say "当前无运行中的 vllm 服务"
  RESTORE=0
fi

kill_all(){
  for PAT in 'vllm serve' 'VLLM::EngineCore' 'Worker_TP' 'from multiprocessing' 'bench_mp' 'benchcap'; do
    $S docker exec vllm bash -c "pkill -9 -f '$PAT' 2>/dev/null" 2>/dev/null
    $S pkill -9 -f "$PAT" 2>/dev/null
  done
  sleep 12
  say "清理后 GPU0 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}')"
}

wait_port(){ P=$1; for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:$P/v1/models 2>/dev/null | grep -q qwen && { say "port $P READY"; return 0; }; sleep 5; done; say "port $P TIMEOUT"; return 1; }

start_pair(){ # $1=盘URI $2=日志前缀
  DISK=$1; LP=$2
  $S bash -c "rm -rf ${DISK#file://}; mkdir -p ${DISK#file://}; chmod 777 ${DISK#file://}"
  for I in 0 1; do
    DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7")
    PORT=$((8000+I))
    $S docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO \
LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
LMCACHE_LOCAL_DISK=$DISK LMCACHE_MAX_LOCAL_DISK_SIZE=1000; \
vllm serve $MODEL --served-model-name qwen \
--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
--port $PORT > ${LP}_i$I.log 2>&1"
    say "实例$I (卡$DEVS, port $PORT) 已启动加载"
    sleep 30
  done
  wait_port 8000 || return 1
  wait_port 8001 || return 1
  say "双实例就绪; GPU0/4 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}') / $(rocm-smi 2>/dev/null | grep -E '^4 ' | awk '{print $(NF-1)}')"
}

pp_pair(){ # 并行灌入: A=0-23, B=24-47
  T0=$(date +%s)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 populate 530 24 0 > /mnt/ws5000/results/tp4_ppA.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 populate 530 24 24 > /mnt/ws5000/results/tp4_ppB.log 2>&1"
  while [ "$($S docker exec vllm bash -c 'pgrep -f bench_mp | wc -l' 2>/dev/null)" -gt 0 ]; do sleep 10; done
  say "populate 完成, 耗时 $(( $(date +%s)-T0 ))s"
  $S cat /mnt/ws5000/results/tp4_ppA.log /mnt/ws5000/results/tp4_ppB.log 2>/dev/null | tail -2
}

meas_pair(){ # $1=标签 $2=每实例N $3=每实例conc $4=iostat设备
  TAG=$1; N=$2; C=$3; DEV=$4
  sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
  nohup bash -c "iostat -x 1 300 $DEV > /tmp/io_tp4_$TAG.log 2>&1" >/dev/null 2>&1 &
  IOP=$!
  T0=$(date +%s.%N)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 measure 530 $N 0 64 $C > /mnt/ws5000/results/tp4_${TAG}_A.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 $N 24 64 $C > /mnt/ws5000/results/tp4_${TAG}_B.log 2>&1"
  while [ "$($S docker exec vllm bash -c 'pgrep -f bench_mp | wc -l' 2>/dev/null)" -gt 0 ]; do sleep 3; done
  T1=$(date +%s.%N)
  sleep 2; kill $IOP 2>/dev/null
  D=$(basename $DEV)
  W=$(awk "BEGIN{printf \"%.1f\", $T1-$T0}")
  AGG=$(awk "BEGIN{printf \"%.1f\", 2*$N*64/($T1-$T0)}")
  say "$TAG: 双实例整批=${W}s 聚合输出吞吐=${AGG} tok/s"
  say "$TAG A: $($S tail -1 /mnt/ws5000/results/tp4_${TAG}_A.log 2>/dev/null)"
  say "$TAG B: $($S tail -1 /mnt/ws5000/results/tp4_${TAG}_B.log 2>/dev/null)"
  say "$TAG 盘峰值: $(awk -v X=$D '$1==X{if($3>m)m=$3}END{printf "%.2f GB/s", m/1e6}' /tmp/io_tp4_$TAG.log)  忙窗均值: $(awk -v X=$D '$1==X && $3>500000{c++;s+=$3}END{if(c)printf "%.2f GB/s (%ds)", s/c/1e6, c}' /tmp/io_tp4_$TAG.log)"
}

# ================= Phase 1: AISSD5000 =================
say "########## TP4x2 · AISSD5000 ##########"
kill_all
start_pair file:///mnt/ws5000/lmcache480tp4 /mnt/ws5000/tp4ws || exit 1
pp_pair
say "落盘=$($S du -sh /mnt/ws5000/lmcache480tp4 | cut -f1)"
meas_pair WS_c8x2  8  8  /dev/md0
meas_pair WS_c16x2 16 16 /dev/md0
# --- 跨实例共享演示: B 实例读 A 灌入的会话 0-3 ---
say "--- 跨实例共享演示: 实例B 冷读 实例A 灌入的会话0-3 ---"
MARKB=$($S bash -c "wc -l < /mnt/ws5000/tp4ws_i1.log")
sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
$S docker exec vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 4 0 64 4" 2>/dev/null | tail -1
say "跨实例取证 need_gt0(B日志新增)=$($S bash -c "tail -n +$MARKB /mnt/ws5000/tp4ws_i1.log | grep -acE 'need to load: [1-9]'")"

# ================= Phase 2: 本地 NVMe =================
say "########## TP4x2 · 本地NVMe ##########"
df -h /srv2 | tail -1
kill_all
start_pair file:///srv2/lmcache480tp4_local /mnt/ws5000/tp4loc || exit 1
pp_pair
say "落盘=$($S du -sh /srv2/lmcache480tp4_local | cut -f1)"
meas_pair LOC_c8x2  8  8  /dev/nvme1n1
meas_pair LOC_c16x2 16 16 /dev/nvme1n1

# ================= Phase 3: 恢复 =================
say "########## 恢复现场 ##########"
kill_all
if [ "$RESTORE" = "1" ] && [ -s /mnt/ws5000/_restore_current.sh ]; then
  $S docker exec -d vllm bash -c "$(cat /mnt/ws5000/_restore_current.sh)"
  for i in $(seq 1 80); do curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen && { say SERVICE_RESTORED; break; }; sleep 5; done
fi
rm -f /tmp/.pw /tmp/.ap
say "ALL_DONE_TP4X2"
