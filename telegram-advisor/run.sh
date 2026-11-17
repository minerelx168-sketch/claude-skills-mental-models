#!/usr/bin/env bash
# Start the advisor bot. Reads .env from this directory.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 bot.py
