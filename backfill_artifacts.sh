#!/usr/bin/env bash
# backfill_artifacts.sh — 未処理 artifacts の一括処理スクリプト
#
# 用途:
#   result_analyst の通常バッチで処理されていない artifacts を
#   一括で処理する（report 生成 + NotebookLM bundle 生成）。
#
# 実行方法:
#   cd /home/limonene/ROCm-project/MAGI-agents
#   bash backfill_artifacts.sh [--dry-run]
#
# 注意:
#   - 本番実行は OpenRouter API を呼び出す（未分析 run 1件につき1回）
#   - --dry-run を先に実行して処理対象を確認すること

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[ERROR] venv が見つかりません: $VENV_PYTHON" >&2
    exit 1
fi

echo "[$(date '+%Y%m%d-%H%M%S')] backfill_artifacts.sh 開始"
echo "[$(date '+%Y%m%d-%H%M%S')] 引数: ${*:-（なし）}"
echo ""

"$VENV_PYTHON" "$SCRIPT_DIR/result_analyst.py" "$@"

echo ""
echo "[$(date '+%Y%m%d-%H%M%S')] 完了"
