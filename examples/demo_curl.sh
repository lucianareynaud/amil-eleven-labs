#!/usr/bin/env bash
# Quick smoke test for /process
set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 path/to/audio-file"
  exit 1
fi

AUDIO_FILE="$1"
echo "Sending $AUDIO_FILE to /process..."
curl -X POST http://localhost:8000/process \
     -F "audio=@${AUDIO_FILE}" \
     -H "Accept: application/json"
