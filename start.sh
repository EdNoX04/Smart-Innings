#!/usr/bin/env bash
# SmartInnings — start backend (FastAPI) + frontend (React) with one command.
#
#   ./start.sh           (or:  bash start.sh)
#
# First run sets up the Python venv and installs dependencies automatically.
# Press Ctrl+C once to stop both servers.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
VENV="$BACKEND/.venv"

echo "🏏  SmartInnings launcher"

# ---- backend setup ----
if [ ! -d "$VENV" ]; then
  echo "→ creating Python virtual environment…"
  python3 -m venv "$VENV"
fi
# install backend deps if uvicorn isn't present yet
if [ ! -x "$VENV/bin/uvicorn" ]; then
  echo "→ installing backend dependencies…"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$BACKEND/requirements.txt"
fi

# ---- frontend setup ----
if [ ! -d "$FRONTEND/node_modules" ]; then
  echo "→ installing frontend dependencies (npm install)…"
  (cd "$FRONTEND" && npm install --silent)
fi

# ---- run both, clean up on exit ----
PIDS=()
cleanup() {
  echo ""
  echo "→ shutting down…"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  exit 0
}
trap cleanup INT TERM

echo "→ starting backend  → http://localhost:8000  (docs at /docs)"
( cd "$BACKEND" && "$VENV/bin/uvicorn" app:app --port 8000 ) &
PIDS+=($!)

echo "→ starting frontend → http://localhost:5173"
( cd "$FRONTEND" && npm run dev ) &
PIDS+=($!)

echo ""
echo "✅  Both running. Open http://localhost:5173   (Ctrl+C to stop both)"
wait
