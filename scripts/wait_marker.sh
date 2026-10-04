#!/usr/bin/env bash
# 轮询等待日志出现标记(后台 notify 用)。用法:wait_marker.sh <marker> <logfile> [max_min]
set -u
marker="$1"
log="$2"
maxmin="${3:-240}"
i=0
while [ "$i" -lt "$maxmin" ]; do
  if grep -q "$marker" "$log" 2>/dev/null; then
    echo "FOUND: $marker (after ${i} min)"
    break
  fi
  sleep 60
  i=$((i + 1))
done
echo "=== tail $log ==="
tail -8 "$log"
