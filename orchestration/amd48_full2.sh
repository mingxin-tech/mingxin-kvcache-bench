#!/bin/bash
# 全指标重测: FS共享池(OWN/CROSS) -> WS LocalDisk -> 本地NVMe -> 重算, 全部 TP4x2
say(){ echo "[$(date +%H:%M:%S)] $*"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8

kill_all(){
  for PAT in 'vllm serve' 'VLLM::EngineCore' 'Worker_TP' 'from multiprocessing' 'bench_mp2[.]py'; do
    $S docker exec vllm bash -c "pkill -9 -f '$PAT' 2>/dev/null" 2>/dev/null
    $S pkill -9 -f "$PAT" 2>/dev/null
  done
  sleep 12
  say "清理后 GPU0 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}')"
}
wait_port(){ P=$1; for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:$P/v1/models 2>/dev/null | grep -q qwen && { say "port $P READY"; return 0; }; sleep 5; done; say "port $P TIMEOUT"; return 1; }
wait_bench(){ sleep 10; while [ "$($S docker exec vllm bash -c "pgrep -fc 'bench_mp2[.]py'" 2>/dev/null)" -gt 0 ]; do sleep 3; done; }

start_pair(){ # $1=LMC环境串(空则无LMCache) $2=日志前缀
  ENVS=$1; LP=$2
  for I in 0 1; do
    DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7")
    PORT=$((8000+I))
    if [ -n "$ENVS" ]; then
      KVC="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}'"
    else
      KVC=""
    fi
    $S docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO $ENVS; \
vllm serve $MODEL --served-model-name qwen \
--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
$KVC --port $PORT > ${LP}_i$I.log 2>&1"
    say "实例$I (卡$DEVS, port $PORT) 启动"
    sleep 30
  done
  wait_port 8000 || return 1
  wait_port 8001 || return 1
  say "双实例就绪 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}') / $(rocm-smi 2>/dev/null | grep -E '^4 ' | awk '{print $(NF-1)}')"
}
pp_pair(){ T0=$(date +%s)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8000 populate 530 24 0 > /mnt/ws5000/results/m2_ppA.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8001 populate 530 24 24 > /mnt/ws5000/results/m2_ppB.log 2>&1"
  wait_bench
  say "populate 完成 $(( $(date +%s)-T0 ))s: $($S tail -qn1 /mnt/ws5000/results/m2_ppA.log /mnt/ws5000/results/m2_ppB.log | tr '\n' ' ')"
  sleep 15
}
meas(){ # $1=标签 $2=N $3=conc $4=A起点 $5=B起点 $6=iostat设备
  TAG=$1; N=$2; C=$3; OA=$4; OB=$5; DEV=$6
  sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
  nohup bash -c "iostat -x 1 900 $DEV > /tmp/io_m2_$TAG.log 2>&1" >/dev/null 2>&1 &
  IOP=$!
  T0=$(date +%s.%N)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8000 measure 530 $N $OA 64 $C > /mnt/ws5000/results/m2_${TAG}_A.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8001 measure 530 $N $OB 64 $C > /mnt/ws5000/results/m2_${TAG}_B.log 2>&1"
  wait_bench
  T1=$(date +%s.%N)
  sleep 2; kill $IOP 2>/dev/null
  D=$(basename $DEV)
  say "$TAG: 整批墙钟=$(awk "BEGIN{printf \"%.1f\",$T1-$T0}")s"
  say "$TAG A: $($S grep -a '^\[p8000\]' /mnt/ws5000/results/m2_${TAG}_A.log | tail -1)"
  say "$TAG B: $($S grep -a '^\[p8001\]' /mnt/ws5000/results/m2_${TAG}_B.log | tail -1)"
  say "$TAG 盘($D): $(awk -v X=$D '$1==X{if($3>m)m=$3} $1==X && $3>500000{c++;s+=$3} END{printf "峰值%.2f GB/s 忙均%.2f GB/s(%ds)", m/1e6, (c?s/c/1e6:0), c}' /tmp/io_m2_$TAG.log)"
}

# scp 上来的新客户端放到位
$S cp /tmp/bench_mp2.py /mnt/ws5000/bench_mp2.py

# ============ Phase 1: fs:// 共享池 (复用已有 344GB 池) ============
say "########## P1 fs共享池 ##########"
say "池现状: $($S du -sh /mnt/ws5000/kvpool_fs | cut -f1) $($S bash -c 'ls /mnt/ws5000/kvpool_fs | wc -l')文件"
kill_all
FSENV="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 LMCACHE_REMOTE_URL=fs://local:0/mnt/ws5000/kvpool_fs LMCACHE_REMOTE_SERDE=naive"
start_pair "$FSENV" /mnt/ws5000/m2fs || exit 1
meas FS_OWN8    8  8 0  24 /dev/md0
meas FS_OWN16   16 16 0 24 /dev/md0
meas FS_CROSS16 16 16 24 0 /dev/md0

# ============ Phase 2: WS LocalDiskBackend ============
say "########## P2 WS LocalDisk ##########"
kill_all
$S bash -c "rm -rf /mnt/ws5000/lmcache480tp4; mkdir -p /mnt/ws5000/lmcache480tp4; chmod 777 /mnt/ws5000/lmcache480tp4"
WSENV="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache480tp4 LMCACHE_MAX_LOCAL_DISK_SIZE=1000"
start_pair "$WSENV" /mnt/ws5000/m2ws || exit 1
pp_pair
say "落盘=$($S du -sh /mnt/ws5000/lmcache480tp4 | cut -f1)"
meas WS8  8  8  0 24 /dev/md0
meas WS16 16 16 0 24 /dev/md0

# ============ Phase 3: 本地 NVMe ============
say "########## P3 本地NVMe ##########"
df -h /srv2 | tail -1
kill_all
$S bash -c "rm -rf /srv2/lmcache480tp4_local; mkdir -p /srv2/lmcache480tp4_local; chmod 777 /srv2/lmcache480tp4_local"
LOCENV="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 LMCACHE_LOCAL_DISK=file:///srv2/lmcache480tp4_local LMCACHE_MAX_LOCAL_DISK_SIZE=1000"
start_pair "$LOCENV" /mnt/ws5000/m2loc || exit 1
pp_pair
say "落盘=$($S du -sh /srv2/lmcache480tp4_local | cut -f1)"
meas LOC8  8  8  0 24 /dev/nvme1n1
meas LOC16 16 16 0 24 /dev/nvme1n1
$S bash -c "rm -rf /srv2/lmcache480tp4_local" &

# ============ Phase 4: 重算 ============
say "########## P4 重算 ##########"
kill_all
start_pair "" /mnt/ws5000/m2rc || exit 1
meas RC8  8  8  0 24 /dev/md0
meas RC16 16 16 0 24 /dev/md0

kill_all
say "ALL_DONE_FULL2"
