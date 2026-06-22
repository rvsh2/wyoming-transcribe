#!/usr/bin/env bash
# Run two services in one container:
#   - HTTP/UI + speaker-enrollment management API on port 8580 (no ASR model loaded)
#   - Wyoming ASR server on port 10300 (foreground)
# They share the enrollment directory; the Wyoming process picks up enrollment
# changes made via the UI on the next transcription.
set -euo pipefail

UI_PORT="${UI_PORT:-8580}"

echo "Starting enrollment UI / API on 0.0.0.0:${UI_PORT} (no model load)"
python3 server.py --host 0.0.0.0 --port "${UI_PORT}" --no-load-model \
    --language "${LANGUAGE:-pl}" &
UI_PID=$!

# If the UI process dies, take the container down too.
trap 'kill -TERM "${UI_PID}" 2>/dev/null || true' EXIT

echo "Starting Wyoming ASR server"
exec python3 -m cohere_wyoming "$@"
