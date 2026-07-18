#!/bin/bash
D(){ sudo -A docker exec vllm bash -lc "$1"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
LM=/root/LMCache/lmcache

echo "===== 0. 机器是否空闲 ====="
pgrep -af 'vllm serve' | head -3
rocm-smi 2>/dev/null | grep -E '^[04] ' | awk '{print $1, $(NF-1)}'

echo "===== 1. config: remote_url/remote_serde 环境变量接线 ====="
D "grep -n 'remote_url\|remote_serde' $LM/v1/config.py | head -20"

echo "===== 2. fs adapter 注册 ====="
D "grep -n -B3 -A10 '\"fs\"\|fs_adapter\|FSAdapter' $LM/v1/storage_backend/connector/__init__.py | head -50"
D "cat $LM/v1/storage_backend/connector/fs_adapter.py"

echo "===== 3. base_connector: save_chunk_meta 默认值/meta_shapes ====="
D "grep -n -B2 -A12 'save_chunk_meta\|meta_shapes' $LM/v1/storage_backend/connector/base_connector.py | head -60"

echo "===== 4. storage_manager: remote backend 挂载顺序与批量get ====="
D "grep -n 'RemoteBackend\|remote_url' $LM/v1/storage_backend/__init__.py $LM/v1/storage_backend/storage_manager.py | head -20"

echo "===== 5. remote_backend: 异步批量取回与并发度 ====="
D "grep -n -B2 -A10 'batched_get\|def get_blocking\|max_workers\|thread' $LM/v1/storage_backend/remote_backend.py | head -80"
