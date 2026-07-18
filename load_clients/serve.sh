#!/bin/bash
CFG=$1
export HIP_VISIBLE_DEVICES=0 PYTHONHASHSEED=0
COMMON="/mnt/ws5000/models/Qwen2.5-7B-Instruct --served-model-name qwen --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.3 --port 8000"
KVT='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
case $CFG in
  A) vllm serve $COMMON --no-enable-prefix-caching > /mnt/ws5000/vllm_A.log 2>&1 ;;
  C) export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=8
     vllm serve $COMMON --enable-prefix-caching --kv-transfer-config "$KVT" > /mnt/ws5000/vllm_C.log 2>&1 ;;
  D) export LMCACHE_CHUNK_SIZE=256 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=8 LMCACHE_LOCAL_DISK=file:///mnt/ws5000/lmcache LMCACHE_MAX_LOCAL_DISK_SIZE=200
     vllm serve $COMMON --enable-prefix-caching --kv-transfer-config "$KVT" > /mnt/ws5000/vllm_D.log 2>&1 ;;
esac