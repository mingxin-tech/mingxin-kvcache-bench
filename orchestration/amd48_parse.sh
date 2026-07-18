#!/bin/bash
awk '$1=="md0"{if($3>m)m=$3}END{printf "md0_peak %.2f GB/s\n", m/1e6}' /tmp/io_repro16.log
awk '$1=="md0" && $3>200000{s+=$3;c++}END{if(c)printf "md0_busy_avg %.2f GB/s (n=%d x1s)\n", s/c/1e6, c}' /tmp/io_repro16.log
awk '$1=="md0"{s+=$3}END{printf "md0_total %.1f GB\n", s/1e6}' /tmp/io_repro16.log
echo "--- md0 高带宽秒级采样 ---"
awk '$1=="md0" && $3>500000{printf "  %.2f GB/s util=%s%%\n", $3/1e6, $NF}' /tmp/io_repro16.log | head -14
echo "--- r_await (md0) ---"
awk '$1=="md0" && $3>500000{print "  r_await="$6"ms"}' /tmp/io_repro16.log | head -5
