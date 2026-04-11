#!/usr/bin/env bash
# Start the YangtzeDelta web server
set -e

cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "⚠️  .env file not found. Copy .env and fill in your API keys."
  exit 1
fi

echo "🚀 Starting YangtzeDelta Proposal Analyser at http://localhost:8000"
.venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
