#!/bin/bash
# stackchan-mcp gateway entrypoint
# Idempotent: if a stackchan-mcp process is already listening on WS_PORT,
# kill it first so the new Hermes session gets a clean stdio connection.
set -euo pipefail

cd /tmp/stackchan-mcp/gateway

WS_PORT="${WS_PORT:-8765}"

cleanup_port() {
    local pid
    pid=$(ss -tlnp 2>/dev/null | grep ":$WS_PORT " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[run_mcp] Port $WS_PORT held by PID $pid, sending SIGTERM..." >&2
        kill "$pid" 2>/dev/null || true
        # wait for port to be released
        for i in 1 2 3 4 5; do
            sleep 1
            if ! ss -tlnp 2>/dev/null | grep -q ":$WS_PORT "; then
                break
            fi
        done
        # force kill if still lingering
        if ss -tlnp 2>/dev/null | grep -q ":$WS_PORT "; then
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi
}

cleanup_port

echo "[run_mcp] Starting stackchan-mcp gateway..." >&2
exec .venv/bin/python -m stackchan_mcp --no-mdns "$@"
