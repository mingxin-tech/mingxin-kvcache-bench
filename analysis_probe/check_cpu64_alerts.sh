#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
for F in rep64_i0 rep64_i1; do
  echo "== $F 告警计数 =="
  sudo -A grep -ac 'Failed to allocate memory block' /mnt/ws5000/$F.log
  sudo -A grep -ac 'get blocking timeout' /mnt/ws5000/$F.log
  sudo -A grep -ac 'No eviction candidates' /mnt/ws5000/$F.log
done
