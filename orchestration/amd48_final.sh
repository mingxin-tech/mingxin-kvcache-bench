#!/bin/bash
echo "===恢复状态==="
cat /tmp/restore.out 2>/dev/null | tail -2
curl -s -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 150; echo
echo "===WS 忙窗分析==="
awk '$1=="md0" && $3>500000 {c++; s+=$3; if($3>m)m=$3} END {printf "busy_secs=%d avg=%.2f peak=%.2f GB/s\n", c, s/c/1e6, m/1e6}' /tmp/io_kv480_WS480.log
echo "===LOC 忙窗分析==="
awk '$1=="nvme1n1" && $3>500000 {c++; s+=$3; if($3>m)m=$3} END {printf "busy_secs=%d avg=%.2f peak=%.2f GB/s\n", c, s/c/1e6, m/1e6}' /tmp/io_kv480_LOC480.log
echo "===WS 高带宽秒采样==="
awk '$1=="md0" && $3>1000000 {printf "%.2f ", $3/1e6} END {print ""}' /tmp/io_kv480_WS480.log
echo "===LOC 高带宽秒采样==="
awk '$1=="nvme1n1" && $3>1000000 {printf "%.2f ", $3/1e6} END {print ""}' /tmp/io_kv480_LOC480.log
