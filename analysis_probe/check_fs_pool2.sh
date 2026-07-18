#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
echo "--- 错误/警告 ---"
sudo -A grep -aE 'ERROR|Failed|error' /mnt/ws5000/fsws_i0.log | grep -av 'is not tuned' | tail -10 | cut -c1-240
echo "--- Stored 行(写入CPU层) ---"
sudo -A grep -ac 'Stored' /mnt/ws5000/fsws_i0.log
sudo -A grep -am2 'Stored' /mnt/ws5000/fsws_i0.log | cut -c1-200
echo "--- put/remote 相关行 ---"
sudo -A grep -aiE 'put_task|remote put|submit.*put|evict' /mnt/ws5000/fsws_i0.log | tail -5 | cut -c1-200
echo "--- 引擎吞吐(灌入是否在跑) ---"
sudo -A grep -a 'Avg prompt throughput' /mnt/ws5000/fsws_i0.log | tail -2 | cut -c1-200
echo "--- 池内(含隐藏/tmp) ---"
sudo -A bash -c 'ls -la /mnt/ws5000/kvpool_fs | head -5'
echo "--- 容器内视角 ---"
sudo -A docker exec vllm bash -c 'ls /mnt/ws5000/kvpool_fs | wc -l; df -h /mnt/ws5000 | tail -1'
echo "--- config里 remote 生效值 ---"
sudo -A grep -am1 "remote_url" /mnt/ws5000/fsws_i0.log | grep -aoE "remote_url[^,]*" | head -2
sudo -A grep -am1 "Creating LMCacheEngine with config" /mnt/ws5000/fsws_i0.log | grep -aoE "remote[^,]*,[^,]*" | head -4
