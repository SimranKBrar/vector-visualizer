#!/usr/bin/env bash
# Starts the FastAPI backend (:8000) and the Vite frontend (:5173) together.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cleanup() {
  trap - EXIT
  kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(cd "$ROOT/backend" && uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload) &
(cd "$ROOT/frontend" && npm run dev) &

echo "Vector Visualizer"
echo "  frontend  -> http://localhost:5173"
echo "  backend   -> http://127.0.0.1:8000  (docs at /docs)"
echo
wait
