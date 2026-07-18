#!/bin/bash
# FS共享池 并发16档 TPOT发散复测: OWN 8+8 连测4轮 + 32档对照1轮
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

meas(){ TAG=$1; N=$2; C=$3
  sync; echo 3 | $S tee /proc/sys/vm/drop_caches >/dev/null
  nohup bash -c "iostat -x 1 200 /dev/md0 > /tmp/io_rep64_$TAG.log 2>&1" >/dev/null 2>&1 &
  IOP=$!
  T0=$(date +%s.%N)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8000 measure 530 $N 0 64 $C > /mnt/ws5000/results/rep64_${TAG}_A.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp2.py 8001 measure 530 $N 24 64 $C > /mnt/ws5000/results/rep64_${TAG}_B.log 2>&1"
  wait_bench
  T1=$(date +%s.%N)
  sleep 2; kill $IOP 2>/dev/null
  say "$TAG A: $($S grep -a '^\[p8000\]' /mnt/ws5000/results/rep64_${TAG}_A.log | tail -1)"
  say "$TAG B: $($S grep -a '^\[p8001\]' /mnt/ws5000/results/rep64_${TAG}_B.log | tail -1)"
  say "$TAG 盘: $(awk '$1=="md0"{if($3>m)m=$3} $1=="md0" && $3>500000{c++;s+=$3} END{printf "峰值%.2f 忙均%.2f GB/s(%ds)", m/1e6, (c?s/c/1e6:0), c}' /tmp/io_rep64_$TAG.log)"
}

say "########## FS共享池 16档发散复测-CPU64G ##########"
say "池现状: $($S du -sh /mnt/ws5000/kvpool_fs | cut -f1)"
kill_all
FSENV="LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=64 LMCACHE_REMOTE_URL=fs://local:0/mnt/ws5000/kvpool_fs LMCACHE_REMOTE_SERDE=naive"
for I in 0 1; do
  DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7")
  PORT=$((8000+I))
  $S docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0 LMCACHE_LOG_LEVEL=INFO $FSENV; \
vllm serve $MODEL --served-model-name qwen \
--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \
--port $PORT > /mnt/ws5000/rep64_i$I.log 2>&1"
  say "实例$I 启动"
  sleep 30
done
wait_port 8000 || exit 1
wait_port 8001 || exit 1
say "双实例就绪"

meas C1_c8x2 8 8
meas C2_c8x2 8 8
meas C3_c8x2 8 8
meas C4_c8x2 8 8
meas C5_c16x2 16 16

kill_all
say "ALL_DONE_FSREP64"
