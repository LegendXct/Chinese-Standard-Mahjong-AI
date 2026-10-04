#!/usr/bin/env bash
# RL 主跑停滞看门狗:每 5 分钟检查 train_metrics.csv 的 update 计数,
# 连续 4 次(20 分钟)无前进则向 rl_main.log 追加 [WATCHDOG] STALL DETECTED。
# 只报警不自动重启(误杀风险大于收益)。报警行附诊断快照:
# 主进程存活 + 子进程数(multiprocessing spawn 的子进程命令行不含模块名,
# 按 pgrep -f 模式数 infer_server/env_worker 恒为 0,故以"train_rl 子进程计数"
# 代替:正常应为 1 infer + 22 worker + 评测若干 ≈ 23+)。
set -u
CSV=/root/autodl-tmp/mahjong/checkpoints/rl_v1/train_metrics.csv
LOG=/root/autodl-tmp/mahjong/logs/rl_main.log
INTERVAL=300
LIMIT=4

last=""
stall=0
while true; do
    upd=$(tail -1 "$CSV" 2>/dev/null | cut -d, -f2)
    if [ -n "$upd" ] && [ "$upd" = "$last" ]; then
        stall=$((stall + 1))
    else
        stall=0
    fi
    last="$upd"
    if [ "$stall" -ge "$LIMIT" ]; then
        # 注意:tee/bash 包装进程的 cmdline 同样含 rl/train_rl.py,
        # 必须按 comm 过滤出真正的 python 主进程,否则 children 计数失真
        tpid=$(for p in $(pgrep -f 'rl/train_rl.py' || true); do
                   [ "$(cat /proc/$p/comm 2>/dev/null)" = "python" ] && echo "$p"
               done | head -1)
        if [ -n "$tpid" ]; then
            alive="alive(pid=$tpid)"
            nchild=$(pgrep -cP "$tpid" || true)
        else
            alive="DEAD"
            nchild=0
        fi
        echo "[WATCHDOG] STALL DETECTED $(date '+%F %T') last_update=${upd:-NA}" \
             "train_rl=$alive children=${nchild}" \
             "patterns: infer_server=$(pgrep -cf infer_server || true)" \
             "env_worker=$(pgrep -cf env_worker || true)" >> "$LOG"
        stall=0   # 重置:若 20 分钟后仍停滞,再报一行
    fi
    sleep "$INTERVAL"
done
