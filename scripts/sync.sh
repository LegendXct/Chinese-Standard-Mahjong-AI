#!/usr/bin/env bash
# 把本地项目同步到 autodl:/root/autodl-tmp/mahjong/
# 用法:
#   scripts/sync.sh              日常增量(不传 data.txt)
#   scripts/sync.sh --with-data  首次全量,包含 200MB 的 data.txt,并做 md5 校验
# 本地是 Windows(Git Bash),无 rsync,采用 tar over ssh。
set -euo pipefail
cd "$(dirname "$0")/.."

HOST=autodl
DEST=/root/autodl-tmp/mahjong
DATA=baseline/SL/data/data.txt

EXCLUDES=(
    --exclude=.git
    --exclude=__pycache__
    --exclude='*.pyc'
    --exclude='*.npz'
    --exclude='*.pt'
    --exclude='*.pkl'
    --exclude='~$*'
)
WITH_DATA=0
if [ "${1:-}" = "--with-data" ]; then
    WITH_DATA=1
else
    EXCLUDES+=(--exclude="$DATA")
fi

ssh "$HOST" "mkdir -p $DEST"
echo "[sync] tar over ssh -> $HOST:$DEST (with_data=$WITH_DATA)"
tar czf - "${EXCLUDES[@]}" . | ssh "$HOST" "tar xzf - -C $DEST"

if [ "$WITH_DATA" = 1 ]; then
    echo "[sync] verifying data.txt md5..."
    LOCAL_MD5=$(md5sum "$DATA" | cut -d' ' -f1)
    REMOTE_MD5=$(ssh "$HOST" "md5sum $DEST/$DATA" | cut -d' ' -f1)
    if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
        echo "[sync] data.txt md5 ok: $LOCAL_MD5"
    else
        echo "[sync] ERROR: data.txt md5 mismatch! local=$LOCAL_MD5 remote=$REMOTE_MD5" >&2
        exit 1
    fi
fi
echo "[sync] done."
