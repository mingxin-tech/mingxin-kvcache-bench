#!/bin/bash
# 启动8实例 32B + LMCache(WS5000 disk backend) 每卡一个,端口8000-8007
BACK=$1   # ws | local
if [ "$BACK" = "ws" ]; then
  MODEL=/mnt/ws5000/models/Qwen2.5-32B-Instruct
  DISK=file:///mnt/ws5000/lmcache_sla
  LOGD=/mnt/ws5000/sla_logs
else
  MODEL=/srv2/models/Qwen2.5-32B-Instruct
  DISK=file:///srv2/lmcache_sla
  LOGD=/mnt/ws5000/sla_logs
fi
mkdir -p $LOGD
pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 5
KVT='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
for i in $(seq 0 7); do
  PORT=$((8000+i))
  HIP_VISIBLE_DEVICES=$i \
  LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
  LMCACHE_LOCAL_DISK=$DISK LMCACHE_MAX_LOCAL_DISK_SIZE=2000 \
  nohup vllm serve $MODEL --served-model-name qwen --dtype bfloat16 \
    --max-model-len 32768 --gpu-memory-utilization 0.9 --port $PORT \
    --enable-prefix-caching --kv-transfer-config "$KVT" \
    > $LOGD/inst_${BACK}_$i.log 2>&1 &
  echo "launched inst $i port $PORT gpu $i"
  sleep 12
done
echo "=== waiting for all 8 ready ==="
for i in $(seq 0 7); do
  for t in $(seq 1 100); do
    if grep -qa "Application startup complete" $LOGD/inst_${BACK}_$i.log 2>/dev/null; then echo "inst $i READY"; break; fi
    if grep -qaE "EngineDeadError|Engine core initialization failed|raise ValueError" $LOGD/inst_${BACK}_$i.log 2>/dev/null; then echo "inst $i FAILED"; break; fi
    sleep 3
  done
done
echo ALL_DONE