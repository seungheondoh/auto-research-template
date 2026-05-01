#!/usr/bin/env bash
# PostToolUse[Write] hook — auto-stages discussion files after save.
set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except Exception:
    print('')
" 2>/dev/null || echo "")

if [[ "$FILE_PATH" == *".research/discussions/"* && ( "$FILE_PATH" == *.json || "$FILE_PATH" == *.md ) ]]; then
    cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    git add "$FILE_PATH" 2>/dev/null || true
    echo ""
    echo "────────────────────────────────────────────────"
    echo "[discuss hook] Discussion staged for commit:"
    echo "  $FILE_PATH"
    echo "  Run: git commit -m 'discuss: <idea summary>'"
    echo "────────────────────────────────────────────────"
fi
