#!/bin/bash
D(){ sudo -A docker exec vllm bash -lc "$1"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
LM=/root/LMCache/lmcache

echo "===== fs_connector 全文 ====="
D "cat $LM/v1/storage_backend/connector/fs_connector.py"

echo "===== remote_backend contains/get 关键路径 ====="
D "grep -n -A8 'def contains\|def exists\|async def get\|def support_ping' $LM/v1/storage_backend/remote_backend.py | head -80"

echo "===== serde（remote 存储格式是否自带 shape/dtype 头） ====="
D "ls $LM/v1/storage_backend/naive_serde/; grep -n -A20 'def serialize' $LM/v1/storage_backend/naive_serde/naive_serde.py 2>/dev/null | head -40"

echo "===== p2p_backend 简介 ====="
D "sed -n '1,50p' $LM/v1/storage_backend/p2p_backend.py"
