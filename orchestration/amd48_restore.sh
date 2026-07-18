#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
$S docker exec -d vllm bash -c "VLLM_ROCM_USE_AITER=1 vllm serve /srv2/Qwen3-Coder-480B-FP8 --served-model-name q3c --tensor-parallel-size 8 --enable-expert-parallel --trust-remote-code --max-model-len 32768 --gpu-memory-utilization 0.9 > /mnt/ws5000/l480v2_local.log 2>&1"
echo "RESTORE_LAUNCHED（加载约2.5分钟）"
for i in $(seq 1 60); do
  curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q q3c && { echo RESTORED_OK; break; }
  sleep 5
done
rm -f /tmp/.pw /tmp/.ap
