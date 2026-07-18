#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
sudo -A docker exec vllm bash -c "grep -n -A60 'def parse_remote_url' /root/LMCache/lmcache/v1/storage_backend/connector/__init__.py | head -80"
echo "===== 容器内直接单测 URL 解析与 FSConnector 建盘 ====="
sudo -A docker exec vllm bash -c "cd /root/LMCache && python3 - <<'EOF'
from lmcache.v1.storage_backend.connector import parse_remote_url
for u in ['fs:///mnt/ws5000/kvpool_fs','fs://local:0/mnt/ws5000/kvpool_fs','fs://localhost/mnt/ws5000/kvpool_fs']:
    try:
        p=parse_remote_url(u)
        print('OK  ', u, '-> host=%r port=%r path=%r'%(p.host,p.port,p.path))
    except Exception as e:
        print('FAIL', u, '->', e)
EOF"
