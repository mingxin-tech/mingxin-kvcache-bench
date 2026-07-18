#!/bin/bash
# 清理残留 GPU 进程（旧 480B 服务的 worker 没被第一次 pkill 匹配到）
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw; printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
pkill -9 -f kv480b.sh 2>/dev/null
$S pkill -9 -f 'vllm serve' 2>/dev/null
$S pkill -9 -f 'VLLM::EngineCore' 2>/dev/null
$S pkill -9 -f 'Worker_TP' 2>/dev/null
$S pkill -9 -f 'from multiprocessing' 2>/dev/null
sleep 12
echo "===清理后 GPU 显存==="
rocm-smi 2>/dev/null | grep -E '^[0-7] ' | awk '{print "GPU"$1, "VRAM%:", $(NF-1)}'
echo "===残留GPU进程==="
$S docker exec vllm bash -c "rocm-smi --showpids 2>/dev/null | grep -c UNKNOWN" 2>/dev/null || true
rm -f /tmp/.pw /tmp/.ap
echo CLEAN_DONE
