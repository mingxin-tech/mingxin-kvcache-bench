#!/bin/bash
echo "${SUDO_PW:?set SUDO_PW env var}" > /tmp/.pw
{ echo '#!/bin/sh'; echo 'cat /tmp/.pw'; } > /tmp/.ap
chmod +x /tmp/.ap
export SUDO_ASKPASS=/tmp/.ap
for F in tp4ws_i0 tp4ws_i1 tp4loc_i0 tp4loc_i1; do
  N1=$(sudo -A grep -acE 'need to load: (2[0-9]{4}|30[0-9]{3})' /mnt/ws5000/$F.log)
  N2=$(sudo -A grep -acE 'hit tokens: 30[0-9]{3}' /mnt/ws5000/$F.log)
  echo "$F need30k=$N1 hit30k=$N2"
done
echo "--- 跨实例4请求在B日志的hit分布 ---"
sudo -A grep -aE 'hit tokens: [0-9]+' /mnt/ws5000/tp4ws_i1.log | grep -aoE 'hit tokens: [0-9]+' | sort | uniq -c | sort -rn | head -6
echo "--- rc 日志确认无 LMCache ---"
sudo -A grep -ac 'LMCache' /mnt/ws5000/tp4rc_i0.log
