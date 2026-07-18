#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
sudo -A docker exec vllm bash -c "sed -n '360,470p' /root/LMCache/lmcache/v1/storage_backend/connector/__init__.py"
echo "===== remote_backend 建连处 ====="
sudo -A docker exec vllm bash -c "sed -n '120,165p' /root/LMCache/lmcache/v1/storage_backend/remote_backend.py"
