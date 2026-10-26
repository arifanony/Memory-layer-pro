#!/bin/bash
# Stop the project Postgres + Redis started by setup-services.sh.
set -u
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PGBIN="$(ls -d "$PROJECT_DIR"/.venv/lib/python*/site-packages/pgserver/pginstall/bin)"
export PATH="$PGBIN:$PATH"
pg_ctl -D "$PROJECT_DIR/.tools/pgdata" stop 2>/dev/null || true
"$PROJECT_DIR/.tools/redis-bin/redis-cli" -p 6379 shutdown nosave 2>/dev/null || true
echo "services stopped"
