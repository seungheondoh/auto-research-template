#!/usr/bin/env bash
# Set up environment variables for Slack, Obsidian, and Anthropic integrations.
# Usage: source scripts/setup_env.sh

# Slack Incoming Webhook URL — paste yours here
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-https://hooks.slack.com/services/T.../B.../xxx}"

# Obsidian git-sync repo path
export OBSIDIAN_VAULT_PATH="${OBSIDIAN_VAULT_PATH:-/workspace/obsidian-sync}"

# Anthropic API key (for multi-agent orchestration scripts)
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-ant-...}"

echo "[setup_env] SLACK_WEBHOOK_URL  = ${SLACK_WEBHOOK_URL:0:50}..."
echo "[setup_env] OBSIDIAN_VAULT_PATH = $OBSIDIAN_VAULT_PATH"
echo "[setup_env] ANTHROPIC_API_KEY  = ${ANTHROPIC_API_KEY:0:20}..."
