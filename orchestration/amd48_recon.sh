#!/bin/bash
export SUDO_ASKPASS=/tmp/.ap
S="sudo -A"
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
printf '#!/bin/sh\ncat /tmp/.pw\n' > /tmp/.ap; chmod +x /tmp/.ap
echo ===GPU_PIDS===
$S docker exec vllm bash -c "rocm-smi --showpids 2>/dev/null | head -20"
echo ===HOST_GPU_PROCS===
ps aux | grep -iE 'python|vllm' | grep -v grep | head -8
echo ===CONTAINER_PROCS===
$S docker exec vllm bash -c "ps aux | head -12"
echo ===KFD===
$S lsof /dev/kfd 2>/dev/null | head -8
