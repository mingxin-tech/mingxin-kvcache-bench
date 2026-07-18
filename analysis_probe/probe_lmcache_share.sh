#!/bin/bash
# 探查 LMCache 源码：本地盘索引机制 + 可用的共享后端
D(){ sudo -A docker exec vllm bash -lc "$1"; }
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap

LM=$(D "python3 -c 'import lmcache,os;print(os.path.dirname(lmcache.__file__))'" | tr -d '\r')
echo "LM_PATH=$LM"

echo "===== 1. LocalDiskBackend 索引结构（init/insert/contains） ====="
D "sed -n '1,80p' $LM/v1/storage_backend/local_disk_backend.py"
echo "----- contains / insert_key -----"
D "grep -n -A6 'def contains\|def insert_key\|self.dict' $LM/v1/storage_backend/local_disk_backend.py | head -60"

echo "===== 2. 磁盘文件命名与元数据来源 ====="
D "grep -n -B2 -A8 'def _key_to_path\|def save_bytes\|DiskCacheMetadata' $LM/v1/storage_backend/local_disk_backend.py | head -70"

echo "===== 3. 可用的 remote connector 类型 ====="
D "ls $LM/v1/storage_backend/connector/ 2>/dev/null"
D "grep -rn 'parse_remote_url\|://' $LM/v1/storage_backend/connector/__init__.py 2>/dev/null | head -30"

echo "===== 4. 是否有 fs/挂载型 connector 或 p2p/controller ====="
D "ls $LM/v1/ | head -30"
D "ls $LM/v1/storage_backend/"
D "grep -rln 'weka\|fsconnector\|filesystem' $LM/v1/storage_backend/ | head"

echo "===== 5. GdsBackend 是否做目录扫描(对照) ====="
D "grep -n -A10 'def __init__' $LM/v1/storage_backend/gds_backend.py 2>/dev/null | head -40"
