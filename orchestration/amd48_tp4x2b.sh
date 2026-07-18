#!/bin/bash
# TP4x2 续跑: WS 测量 -> 跨实例演示 -> 本地NVMe 全流程
say(){ echo "[$(date +%H:%M:%S)] $*"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8

kill_all(){
  for PAT in 'vllm serve' 'VLLM::EngineCore' 'Worker_TP' 'from multiprocessing' 'bench_mp[.]py' 'benchcap'; do
    $S docker exec vllm bash -c "pkill -9 -f '$PAT' 2>/dev/null" 2>/dev/null
    $S pkill -9 -f "$PAT" 2>/dev/null
  done
  sleep 12
  say "清理后 GPU0 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}')"
}
wait_port(){ P=$1; for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:$P/v1/models 2>/dev/null | grep -q qwen && { say "port $P READY"; return 0; }; sleep 5; done; say "port $P TIMEOUT"; return 1; }
wait_bench(){ while [ "$($S docker exec vllm bash -c "pgrep -fc 'bench_mp[.]py'" 2>/dev/null)" -gt 0 ]; do sleep 3; done; }

start_pair(){ DISK=$1; LP=$2
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
    say "实例$I (卡$DEVS, port $PORT) 启动"
    sleep 30
  done
  wait_port 8000 || return 1
  wait_port 8001 || return 1
  say "双实例就绪 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}') / $(rocm-smi 2>/dev/null | grep -E '^4 ' | awk '{print $(NF-1)}')"
}
pp_pair(){ T0=$(date +%s)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 populate 530 24 0 > /mnt/ws5000/results/tp4_ppA.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 populate 530 24 24 > /mnt/ws5000/results/tp4_ppB.log 2>&1"
  sleep 20; wait_bench
  say "populate 完成 $(( $(date +%s)-T0 ))s: $($S tail -qn1 /mnt/ws5000/results/tp4_ppA.log /mnt/ws5000/results/tp4_ppB.log | tr '\n' ' ')"
}
meas_pair(){ TAG=$1; N=$2; C=$3; DEV=$4
  sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
  nohup bash -c "iostat -x 1 400 $DEV > /tmp/io_tp4_$TAG.log 2>&1" >/dev/null 2>&1 &
  IOP=$!
  T0=$(date +%s.%N)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 measure 530 $N 0 64 $C > /mnt/ws5000/results/tp4_${TAG}_A.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 $N 24 64 $C > /mnt/ws5000/results/tp4_${TAG}_B.log 2>&1"
  sleep 10; wait_bench
  T1=$(date +%s.%N)
  sleep 2; kill $IOP 2>/dev/null
  D=$(basename $DEV)
  say "$TAG: 整批墙钟=$(awk "BEGIN{printf \"%.1f\",$T1-$T0}")s 聚合吞吐=$(awk "BEGIN{printf \"%.1f\",2*$N*64/($T1-$T0)}") tok/s"
  say "$TAG A: $($S tail -1 /mnt/ws5000/results/tp4_${TAG}_A.log)"
  say "$TAG B: $($S tail -1 /mnt/ws5000/results/tp4_${TAG}_B.log)"
  say "$TAG 盘: $(awk -v X=$D '$1==X{if($3>m)m=$3} $1==X && $3>500000{c++;s+=$3} END{printf "峰值%.2f GB/s 忙均%.2f GB/s(%ds)", m/1e6, (c?s/c/1e6:0), c}' /tmp/io_tp4_$TAG.log)"
}

# ============ Phase 1 续: AISSD5000 测量 ============
say "########## WS 测量 ##########"
meas_pair WS_c8x2  8  8  /dev/md0
meas_pair WS_c16x2 16 16 /dev/md0
say "--- 跨实例共享: 实例B 冷读 实例A 灌的会话0-3 ---"
MARKB=$($S bash -c "wc -l < /mnt/ws5000/tp4ws_i1.log")
sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
$S docker exec vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 4 0 64 4" | tail -1
say "跨实例 need_gt0(B新增日志)=$($S bash -c "tail -n +$MARKB /mnt/ws5000/tp4ws_i1.log | grep -acE 'need to load: [1-9]' ") hit日志=$($S bash -c "tail -n +$MARKB /mnt/ws5000/tp4ws_i1.log | grep -acE 'Reqid.*hit' ")"

# ============ Phase 2: 本地 NVMe ============
say "########## TP4x2 本地NVMe ##########"
kill_all
start_pair file:///srv2/lmcache480tp4_local /mnt/ws5000/tp4loc || exit 1
pp_pair
say "落盘=$($S du -sh /srv2/lmcache480tp4_local | cut -f1)"
meas_pair LOC_c8x2  8  8  /dev/nvme1n1
meas_pair LOC_c16x2 16 16 /dev/nvme1n1

# ============ 收尾 ============
say "########## 收尾清理 ##########"
kill_all
$S bash -c "rm -rf /srv2/lmcache480tp4_local"
say "ALL_DONE_TP4X2B"
