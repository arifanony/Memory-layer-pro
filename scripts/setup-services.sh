#!/bin/bash
# One-time + every-boot service setup for Memory Layer Pro (run in YOUR terminal,
# not in the sandboxed agent: the sandbox cannot bind ports or look up uids).
# Idempotent: safe to re-run. Uses only project-local binaries and data dirs.
set -u
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PGBIN="$(ls -d "$PROJECT_DIR"/.venv/lib/python*/site-packages/pgserver/pginstall/bin)"
export PATH="$PGBIN:$PATH"

echo "== Postgres 16 (project cluster) =="
if [ ! -f "$PROJECT_DIR/.tools/pgdata/PG_VERSION" ]; then
  initdb -D "$PROJECT_DIR/.tools/pgdata" -U postgres --auth=trust -E UTF8
fi
if ! pg_isready -h localhost -p 5432 -q; then
  pg_ctl -D "$PROJECT_DIR/.tools/pgdata" -l "$PROJECT_DIR/.tools/pg.log" \
    -o "-p 5432 -k /tmp" start
fi

echo "== pgvector extension =="
if [ ! -d "$PROJECT_DIR/.tools/pgvector-src" ]; then
  git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git \
    "$PROJECT_DIR/.tools/pgvector-src"
fi
make -C "$PROJECT_DIR/.tools/pgvector-src" PG_CONFIG="$PGBIN/pg_config" >/dev/null \
  && make -C "$PROJECT_DIR/.tools/pgvector-src" PG_CONFIG="$PGBIN/pg_config" install
createdb -h localhost -p 5432 -U postgres memory_lab 2>/dev/null || true
psql -h localhost -p 5432 -U postgres -d memory_lab -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "== Redis 8 (project binary) =="
if ! "$PROJECT_DIR/.tools/redis-bin/redis-cli" -p 6379 ping >/dev/null 2>&1; then
  "$PROJECT_DIR/.tools/redis-bin/redis-server" --port 6379 \
    --dir "$PROJECT_DIR/.tools/redis-data" --save '' --appendonly no \
    --daemonize yes --logfile "$PROJECT_DIR/.tools/redis.log"
fi

echo "== Verify =="
pg_isready -h localhost -p 5432
"$PROJECT_DIR/.tools/redis-bin/redis-cli" -p 6379 ping
psql -h localhost -p 5432 -U postgres -d memory_lab -tAc "SELECT extversion FROM pg_extension WHERE extname='vector';"
