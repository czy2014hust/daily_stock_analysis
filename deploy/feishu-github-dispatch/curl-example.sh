#!/usr/bin/env bash
# Test GitHub repository_dispatch from your Mac (no Cloudflare).
# Usage:
#   export GITHUB_TOKEN=github_pat_xxx
#   export GITHUB_REPO=czy2014hust/daily_stock_analysis
#   ./deploy/feishu-github-dispatch/curl-example.sh "算力板块量化策略"

set -euo pipefail

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "Set GITHUB_TOKEN first (fine-grained PAT with Actions: Write, or classic repo scope)" >&2
  exit 2
fi

REPO="${GITHUB_REPO:-czy2014hust/daily_stock_analysis}"
TEXT="${1:-curl 连通性测试：只验证能否触发 Actions}"

BODY=$(python3 -c 'import json,os,sys; print(json.dumps({"event_type":"feishu-strategy-research","client_payload":{"text":sys.argv[1]}},ensure_ascii=False))' "$TEXT")

echo "POST https://api.github.com/repos/${REPO}/dispatches"
echo "text: $TEXT"

curl -sS -o /tmp/gh-dispatch-body.txt -w "HTTP %{http_code}\n" \
  -X POST "https://api.github.com/repos/${REPO}/dispatches" \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -H "Content-Type: application/json" \
  -d "$BODY"

if [ -s /tmp/gh-dispatch-body.txt ]; then
  echo "Response body:"
  cat /tmp/gh-dispatch-body.txt
  echo
fi

echo "Expect HTTP 204. Then open: https://github.com/${REPO}/actions"
