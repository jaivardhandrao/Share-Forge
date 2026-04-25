#!/usr/bin/env bash
#
# validate-submission.sh - OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#
# Run:
#   ./pre_validation.py <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
stop_at() { printf "\n${RED}${BOLD}Stopped at %s.${NC}\n" "$1"; exit 1; }

printf "\n${BOLD}Share-Forge Submission Validator${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"

log "${BOLD}Step 1/3: Pinging HF Space${NC}"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 || printf "000")
if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space responds to /reset"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: docker build${NC}"
if ! command -v docker &>/dev/null; then
  fail "docker not installed"
  stop_at "Step 2"
fi
if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$REPO_DIR" >/dev/null 2>&1; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: openenv validate${NC}"
if ! command -v openenv &>/dev/null; then
  fail "openenv not installed (pip install openenv-core)"
  stop_at "Step 3"
fi
if (cd "$REPO_DIR" && openenv validate) >/dev/null 2>&1; then
  pass "openenv validate passed"
else
  fail "openenv validate failed"
  stop_at "Step 3"
fi

printf "\n${GREEN}${BOLD}All 3/3 checks passed.${NC}\n"
exit 0
