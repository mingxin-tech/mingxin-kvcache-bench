#!/bin/bash
# 方案一验证：fs:// remote connector 跨实例热共享 + 性能同口径
say(){ echo "[$(date +%H:%M:%S)] $*"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8
POOL=/mnt/ws5000/kvpool_fs

kill_all(){
  for PAT in 'vllm serve' 'VLLM::EngineCore' 'Worker_TP' 'from multiprocessing' 'bench_mp[.]py'; do
    $S docker exec vllm bash -c "pkill -9 -f '$PAT' 2>/dev/null" 2>/dev/null
    $S pkill -9 -f "$PAT" 2>/dev/null
  done
  sleep 12
  say "清理后 GPU0 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}')"
}
wait_port(){ P=$1; for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:$P/v1/models 2>/dev/null | grep -q qwen && { say "port $P READY"; return 0; }; sleep 5; done; say "port $P TIMEOUT"; return 1; }
wait_bench(){ while [ "$($S docker exec vllm bash -c "pgrep -fc 'bench_mp[.]py'" 2>/dev/null)" -gt 0 ]; do sleep 3; done; }

say "########## 方案一: fs:// 共享池 TP4x2 ##########"
kill_all
$S bash -c "rm -rf /mnt/ws5000/lmcache480tp4 $POOL; mkdir -p $POOL; chmod 777 $POOL"
df -h /mnt/ws5000 | tail -1

for I in 0 1; do
  DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7")
  PORT=$((8000+I))
  $S docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO \
LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
LMCACHE_REMOTE_URL=fs://local:0$POOL LMCACHE_REMOTE_SERDE=naive; \
vllm serve $MODEL --served-model-name qwen \
--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
--port $PORT > /mnt/ws5000/fsws_i$I.log 2>&1"
  say "实例$I (卡$DEVS, port $PORT) 启动"
  sleep 30
done
wait_port 8000 || exit 1
wait_port 8001 || exit 1
say "双实例就绪 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}') / $(rocm-smi 2>/dev/null | grep -E '^4 ' | awk '{print $(NF-1)}')"
say "RemoteConn确认: $($S grep -ac 'Connection initialized' /mnt/ws5000/fsws_i0.log)/$($S grep -ac 'Connection initialized' /mnt/ws5000/fsws_i1.log)"

# ---- 并行灌入 ----
T0=$(date +%s)
$S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 populate 530 24 0 > /mnt/ws5000/results/fs_ppA.log 2>&1"
$S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 populate 530 24 24 > /mnt/ws5000/results/fs_ppB.log 2>&1"
sleep 20; wait_bench
say "populate 完成 $(( $(date +%s)-T0 ))s: $($S tail -qn1 /mnt/ws5000/results/fs_ppA.log /mnt/ws5000/results/fs_ppB.log | tr '\n' ' ')"
sleep 20  # 等异步put队列排空
say "共享池落盘=$($S du -sh $POOL | cut -f1) 文件数=$($S bash -c "ls $POOL | wc -l")"

meas_pair(){ # $1=标签 $2=每实例N $3=conc $4=A起点 $5=B起点
  TAG=$1; N=$2; C=$3; OA=$4; OB=$5
  MK0=$($S bash -c "wc -l < /mnt/ws5000/fsws_i0.log")
  MK1=$($S bash -c "wc -l < /mnt/ws5000/fsws_i1.log")
  sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
  nohup bash -c "iostat -x 1 400 /dev/md0 > /tmp/io_fs_$TAG.log 2>&1" >/dev/null 2>&1 &
  IOP=$!
  T0=$(date +%s.%N)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 measure 530 $N $OA 64 $C > /mnt/ws5000/results/fs_${TAG}_A.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 $N $OB 64 $C > /mnt/ws5000/results/fs_${TAG}_B.log 2>&1"
  sleep 10; wait_bench
  T1=$(date +%s.%N)
  sleep 2; kill $IOP 2>/dev/null
  say "$TAG: 整批墙钟=$(awk "BEGIN{printf \"%.1f\",$T1-$T0}")s 聚合吞吐(按墙钟)=$(awk "BEGIN{printf \"%.1f\",2*$N*64/($T1-$T0)}") tok/s"
  say "$TAG A(off$OA): $($S tail -1 /mnt/ws5000/results/fs_${TAG}_A.log)"
  say "$TAG B(off$OB): $($S tail -1 /mnt/ws5000/results/fs_${TAG}_B.log)"
  say "$TAG 盘: $(awk '$1=="md0"{if($3>m)m=$3} $1=="md0" && $3>500000{c++;s+=$3} END{printf "峰值%.2f GB/s 忙均%.2f GB/s(%ds)", m/1e6, (c?s/c/1e6:0), c}' /tmp/io_fs_$TAG.log)"
  H0=$($S bash -c "tail -n +$MK0 /mnt/ws5000/fsws_i0.log | grep -acE 'hit tokens: 30[0-9]{3}'")
  H1=$($S bash -c "tail -n +$MK1 /mnt/ws5000/fsws_i1.log | grep -acE 'hit tokens: 30[0-9]{3}'")
  Z0=$($S bash -c "tail -n +$MK0 /mnt/ws5000/fsws_i0.log | grep -acE 'hit tokens: 0,'")
  Z1=$($S bash -c "tail -n +$MK1 /mnt/ws5000/fsws_i1.log | grep -acE 'hit tokens: 0,'")
  say "$TAG 取证: A满命中=$H0/$N 零命中=$Z0 | B满命中=$H1/$N 零命中=$Z1"
}

say "===== 档1: 读自己灌的会话(同口径基线) 8+8 ====="
meas_pair OWN_c8x2 8 8 0 24
say "===== 档2: 读自己灌的会话 16+16 ====="
meas_pair OWN_c16x2 16 16 0 24
say "===== 档3: 交叉冷读(热共享证明) A读B灌的24-39, B读A灌的0-15, 16+16 ====="
meas_pair CROSS_c16x2 16 16 24 0

kill_all
say "ALL_DONE_FSSHARE"
