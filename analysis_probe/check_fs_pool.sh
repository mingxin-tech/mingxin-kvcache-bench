#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
echo "--- pool ---"
sudo -A ls /mnt/ws5000/kvpool_fs | wc -l
sudo -A du -sh /mnt/ws5000/kvpool_fs
echo "--- 后端初始化相关日志 ---"
sudo -A grep -aE 'RemoteBackend|FSConnector|remote_url|Creating FS|storage backend|StorageManager' /mnt/ws5000/fsws_i0.log | head -20 | cut -c1-220
echo "--- config dump行 ---"
sudo -A grep -aE 'remote' /mnt/ws5000/fsws_i0.log | head -10 | cut -c1-260
echo "--- populate进度 ---"
sudo -A tail -qn1 /mnt/ws5000/results/fs_ppA.log /mnt/ws5000/results/fs_ppB.log 2>/dev/null
sudo -A docker exec vllm bash -c "pgrep -fc 'bench_mp[.]py'" 2>/dev/null
echo "--- 编排日志 ---"
grep -vE 'Killed' /tmp/fsshare.out | tail -3
