#!/usr/bin/env bash
# Run two services in one container:
#   - HTTP/UI + speaker-enrollment management API on port 8580 (no ASR model loaded)
#   - Wyoming ASR server on port 10300
# They share the enrollment directory; the Wyoming process picks up enrollment
# changes made via the UI on the next transcription.
#
# Both run as background children so this shell can supervise them: when either
# one dies the other is stopped and the container exits, letting Docker's
# restart policy bring the pair back up together.
set -euo pipefail

UI_PORT="${UI_PORT:-8580}"

echo "Starting enrollment UI / API on 0.0.0.0:${UI_PORT} (no model load)"
python3 server.py --host 0.0.0.0 --port "${UI_PORT}" --no-load-model \
    --language "${LANGUAGE:-pl}" &
UI_PID=$!

echo "Starting Wyoming ASR server"
python3 -m cohere_wyoming "$@" &
WYOMING_PID=$!

stop_children() {
    kill -TERM "${UI_PID}" "${WYOMING_PID}" 2>/dev/null || true
}
trap stop_children TERM INT

status=0
wait -n "${UI_PID}" "${WYOMING_PID}" || status=$?
echo "A service exited (status ${status}); shutting down the container"
stop_children
wait || true
exit "${status}"
