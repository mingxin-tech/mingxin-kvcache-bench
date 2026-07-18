#!/bin/bash
# 只读检查同事服务器的 LMCache 现状
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"

echo "===LMCache 版本与安装方式==="
$S docker exec vllm bash -c "pip show lmcache 2>/dev/null | head -6"
$S docker exec vllm bash -c "python3 -c 'import lmcache; print(\"module path:\", lmcache.__file__)' 2>/dev/null"

echo "===源码仓库状态（/root/LMCache）==="
$S docker exec vllm bash -c "cd /root/LMCache 2>/dev/null && git log -3 --format='%h %ad %s' --date=short 2>/dev/null; git -C /root/LMCache branch --show-current 2>/dev/null; git -C /root/LMCache describe --tags 2>/dev/null"
$S docker exec vllm bash -c "git -C /root/LMCache status --porcelain 2>/dev/null | head -10"

echo "===c_ops 构建==="
$S docker exec vllm bash -c "python3 -c 'import lmcache.c_ops; print(\"c_ops OK\")' 2>&1 | tail -1"
$S docker exec vllm bash -c "ls /root/LMCache/lmcache/*.so /root/LMCache/build 2>/dev/null | head -5"

echo "===PAR-READ 补丁状态==="
$S docker exec vllm bash -c "grep -c 'PAR-READ PATCH' /root/LMCache/lmcache/v1/storage_backend/local_disk_backend.py 2>/dev/null"
$S docker exec vllm bash -c "ls /root/LMCache/lmcache/v1/storage_backend/local_disk_backend.py.bak* 2>/dev/null"

echo "===async_loading / layerwise 相关代码==="
$S docker exec vllm bash -c "grep -rn 'enable_async_loading' /root/LMCache/lmcache/v1/config.py 2>/dev/null | head -3"
$S docker exec vllm bash -c "grep -n 'async_lookup_and_prefetch' /root/LMCache/lmcache/v1/storage_backend/storage_manager.py 2>/dev/null | head -3"
echo "--- layerwise 按请求隔离修复(#2613)特征: layerwise storer 是否 request-scoped ---"
$S docker exec vllm bash -c "grep -rn 'req_id' /root/LMCache/lmcache/v1/gpu_connector/*.py 2>/dev/null | grep -i 'layerwise\|storer' | head -5"
$S docker exec vllm bash -c "grep -rn 'layerwise_storers\|save_generators' /root/LMCache/lmcache/v1/ 2>/dev/null | head -8"

echo "===GDS 后端与线程池==="
$S docker exec vllm bash -c "grep -n '_DEFAULT_THREAD_COUNT\|gds_io_threads' /root/LMCache/lmcache/v1/storage_backend/gds_backend.py 2>/dev/null | head -5"

echo "===vLLM 版本==="
$S docker exec vllm bash -c "pip show vllm 2>/dev/null | head -2"

rm -f /tmp/.pw /tmp/.ap
echo "===DONE(只读，凭据已清)==="
