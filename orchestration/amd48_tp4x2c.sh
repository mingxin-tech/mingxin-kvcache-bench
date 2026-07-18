#!/bin/bash
# TP4x2 重算基线轮：双实例、无 LMCache，c8x2 与 c16x2
say(){ echo "[$(date +%H:%M:%S)] $*"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
MODEL=/mnt/ws5000/models/Qwen3-Coder-480B-FP8

kill_all(){
  for PAT in 'vllm serve' 'VLLM::EngineCore' 'Worker_TP' 'from multiprocessing' 'bench_mp[.]py'; do
    $S docker exec vllm bash -c "pkill -9 -f '$PAT' 2>/dev/null" 2>/dev/null
    $S pkill -9 -f "$PAT" 2>/dev/null
  done
  sleep 12
  say "清理后 GPU0 VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}')"
}
wait_port(){ P=$1; for i in $(seq 1 200); do curl -s -m 3 http://127.0.0.1:$P/v1/models 2>/dev/null | grep -q qwen && { say "port $P READY"; return 0; }; sleep 5; done; say "port $P TIMEOUT"; return 1; }
wait_bench(){ while [ "$($S docker exec vllm bash -c "pgrep -fc 'bench_mp[.]py'" 2>/dev/null)" -gt 0 ]; do sleep 5; done; }

kill_all
for I in 0 1; do
  DEVS=$([ $I -eq 0 ] && echo "0,1,2,3" || echo "4,5,6,7")
  PORT=$((8000+I))
  $S docker exec -d vllm bash -c "export HIP_VISIBLE_DEVICES=$DEVS VLLM_ROCM_USE_AITER=1 PYTHONHASHSEED=0; \
vllm serve $MODEL --served-model-name qwen \
--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code \
--max-model-len 32768 --gpu-memory-utilization 0.9 --no-enable-prefix-caching \
--port $PORT > /mnt/ws5000/tp4rc_i$I.log 2>&1"
  say "重算实例$I (卡$DEVS, port $PORT) 启动"
  sleep 30
done
wait_port 8000 || exit 1
wait_port 8001 || exit 1
say "双实例就绪(无LMCache) VRAM: $(rocm-smi 2>/dev/null | grep -E '^0 ' | awk '{print $(NF-1)}') / $(rocm-smi 2>/dev/null | grep -E '^4 ' | awk '{print $(NF-1)}')"

meas(){ TAG=$1; N=$2; C=$3
  T0=$(date +%s.%N)
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8000 measure 530 $N 0 64 $C > /mnt/ws5000/results/tp4_${TAG}_A.log 2>&1"
  $S docker exec -d vllm bash -c "python3 /mnt/ws5000/bench_mp.py 8001 measure 530 $N 24 64 $C > /mnt/ws5000/results/tp4_${TAG}_B.log 2>&1"
  sleep 10; wait_bench
  T1=$(date +%s.%N)
  say "$TAG: 整批墙钟=$(awk "BEGIN{printf \"%.1f\",$T1-$T0}")s 聚合吞吐=$(awk "BEGIN{printf \"%.1f\",2*$N*64/($T1-$T0)}") tok/s"
  say "$TAG A: $($S tail -1 /mnt/ws5000/results/tp4_${TAG}_A.log)"
  say "$TAG B: $($S tail -1 /mnt/ws5000/results/tp4_${TAG}_B.log)"
}
meas RC_c8x2  8  8
meas RC_c16x2 16 16

kill_all
rm -f /tmp/.pw /tmp/.ap
say "ALL_DONE_TP4X2C"
