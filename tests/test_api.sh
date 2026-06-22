#!/bin/bash
# ============================================================================
# Test script for the HTTP debug server
# Compatible with the basic whisper.cpp-style API exposed by server.py
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVER="${SERVER:-http://127.0.0.1:8580}"
DEFAULT_AUDIO="${REPO_ROOT}/Recording.wav"

if [[ $# -ge 1 ]]; then
  AUDIO_FILE="$1"
else
  AUDIO_FILE="${DEFAULT_AUDIO}"
fi

echo "============================================"
echo "  Cohere Transcribe Server - API Tests"
echo "============================================"
echo ""
echo "Server: ${SERVER}"
echo "Audio:  ${AUDIO_FILE}"
echo ""

if [[ ! -f "${AUDIO_FILE}" ]]; then
  echo "Audio file not found: ${AUDIO_FILE}" >&2
  exit 1
fi

# --- Test 1: Server info page ---
echo "=== Test 1: GET / (server info) ==="
curl -sS "${SERVER}/" | head -5
echo ""
echo ""

# --- Test 2: /inference JSON (whisper.cpp format) ---
echo "=== Test 2: POST /inference (json) ==="
curl -sS "${SERVER}/inference" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${AUDIO_FILE}" \
  -F "temperature=0.0" \
  -F "temperature_inc=0.2" \
  -F "response_format=json" \
  -F "language=en"
echo ""
echo ""

# --- Test 3: /inference text format ---
echo "=== Test 3: POST /inference (text) ==="
curl -sS "${SERVER}/inference" \
  -F "file=@${AUDIO_FILE}" \
  -F "response_format=text" \
  -F "language=en"
echo ""

# --- Test 4: /inference verbose_json format ---
echo "=== Test 4: POST /inference (verbose_json) ==="
curl -sS "${SERVER}/inference" \
  -F "file=@${AUDIO_FILE}" \
  -F "response_format=verbose_json" \
  -F "language=en"
echo ""
echo ""

# --- Test 5: /inference SRT format ---
echo "=== Test 5: POST /inference (srt) ==="
curl -sS "${SERVER}/inference" \
  -F "file=@${AUDIO_FILE}" \
  -F "response_format=srt" \
  -F "language=en"
echo ""

# --- Test 6: /inference VTT format ---
echo "=== Test 6: POST /inference (vtt) ==="
curl -sS "${SERVER}/inference" \
  -F "file=@${AUDIO_FILE}" \
  -F "response_format=vtt" \
  -F "language=en"
echo ""

# --- Test 7: OpenAI compatible endpoint ---
echo "=== Test 7: POST /v1/audio/transcriptions (OpenAI) ==="
curl -sS "${SERVER}/v1/audio/transcriptions" \
  -F "file=@${AUDIO_FILE}" \
  -F "model=CohereLabs/cohere-transcribe-03-2026" \
  -F "language=en"
echo ""
echo ""

# --- Test 8: Polish language ---
echo "=== Test 8: POST /inference (Polish) ==="
curl -sS "${SERVER}/inference" \
  -F "file=@${AUDIO_FILE}" \
  -F "response_format=json" \
  -F "language=pl"
echo ""
echo ""

echo "============================================"
echo "  All tests completed!"
echo "============================================"
