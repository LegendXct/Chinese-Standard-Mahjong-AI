#!/usr/bin/env bash
# cgroup 内存哨兵:每 60s 记录,连续 3 次 >85GB 写醒目告警(不杀任务)
LOG="${1:-/root/autodl-tmp/mahjong/logs/mem_sentinel.log}"
over=0
while true; do
    cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    gb=$((cur / 1073741824))
    echo "[mem] $(date +%H:%M:%S) ${gb}GB" >> "$LOG"
    if [ "$gb" -gt 85 ]; then
        over=$((over + 1))
        [ "$over" -ge 3 ] && echo "[mem] !!!! WARNING >85GB x$over $(date) !!!!" >> "$LOG"
    else
        over=0
    fi
    sleep 60
done
