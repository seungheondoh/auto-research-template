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

if [[ "$FILE_PATH" == */research/hypotheses/*.md || "$FILE_PATH" == */research/experiments/*.md || "$FILE_PATH" == */research/log.md ]]; then
    cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    git add "$FILE_PATH" 2>/dev/null || true
    echo ""
    echo "────────────────────────────────────────────────"
    echo "[research hook] Research artifact staged:"
    echo "  $FILE_PATH"
    echo "  Run: git commit -m 'research: <summary>'"
    echo "────────────────────────────────────────────────"
fi
