#!/usr/bin/env sh
# Restore a backup taken by ops/backup.sh into the current stack.
#
#   ops/restore.sh backups/20260818T101500Z
#
# The drill this is written for restores into a *clean* stack — the point is to
# find out what the backup does not contain, and a stack that still has the
# original data cannot tell you that. So this refuses to run against a database
# that already has runs in it unless FORCE=1 is set.
#
# Afterwards, run ops/verify_restore.py. "The restore command exited zero" and
# "a run from before still opens and still predicts" are different claims.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_ROOT"

# See the same block in backup.sh: without this, Git Bash rewrites the
# container-side /backup and /models paths into Windows paths and the restore
# silently reads from, or writes to, the wrong place.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

SRC=${1:?usage: ops/restore.sh <backup directory>}
[ -d "$SRC" ] || { echo "No such backup: $SRC" >&2; exit 1; }

POSTGRES_USER=${POSTGRES_USER:-analyst}
POSTGRES_DB=${POSTGRES_DB:-analyst}
S3_BUCKET=${S3_BUCKET:-analyst-artifacts}

# Same reasoning as backup.sh: restore into the database the application reads,
# not the one the server happened to create. Restoring into the wrong database
# leaves a stack that starts cleanly and has lost everything.
case "${DATABASE_URL:-}" in
  postgres*)
    _rest=${DATABASE_URL#*://}
    _creds=${_rest%%@*}
    _db=${_rest##*/}
    _db=${_db%%\?*}
    [ -n "$_db" ] && POSTGRES_DB=$_db
    [ -n "${_creds%%:*}" ] && POSTGRES_USER=${_creds%%:*}
    ;;
esac

running() {
  docker compose ps --services --status running 2>/dev/null | grep -qx "$1"
}

EXISTING=$(curl -fsS http://localhost:8000/api/runs 2>/dev/null \
  | python -c "import json,sys; print(json.load(sys.stdin)['count'])" 2>/dev/null || echo 0)
if [ "${EXISTING:-0}" -gt 0 ] && [ "${FORCE:-0}" != "1" ]; then
  echo "This stack already has $EXISTING run(s)." >&2
  echo "Restore into a clean stack (docker compose down -v && docker compose up -d)," >&2
  echo "or set FORCE=1 if you really mean to overwrite this one." >&2
  exit 1
fi

echo "Restoring from $SRC"

# --- the database ------------------------------------------------------------
if [ -f "$SRC/database.dump" ]; then
  running postgres || { echo "This backup is Postgres; start the postgres profile first." >&2; exit 1; }
  echo "  postgres: pg_restore --clean"
  # --clean --if-exists so a partially-restored database is recoverable by
  # running this again, rather than failing on objects that already exist.
  docker compose exec -T postgres \
    pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner \
    < "$SRC/database.dump"
elif [ -f "$SRC/analyst.db" ]; then
  echo "  sqlite: replacing models/analyst.db"
  # Stop the API first: overwriting the file underneath an open connection
  # leaves the running process reading a database that no longer exists.
  docker compose stop backend worker >/dev/null 2>&1 || true
  docker compose run --rm -T --entrypoint sh backend -c \
    'cat > /app/models/analyst.db && rm -f /app/models/analyst.db-wal /app/models/analyst.db-shm' \
    < "$SRC/analyst.db"
  # Both, not just the API: the worker was stopped too, and a stack whose
  # worker never came back looks like a queue that accepts and never runs.
  docker compose start backend worker >/dev/null 2>&1 || true
else
  echo "Backup has neither database.dump nor analyst.db" >&2
  exit 1
fi

# --- the artifacts -----------------------------------------------------------
if [ -d "$SRC/bucket" ]; then
  running minio || { echo "This backup has bucket contents; start the storage profile first." >&2; exit 1; }
  echo "  minio: mirroring back into s3://$S3_BUCKET"
  NETWORK=$(docker compose ps --format '{{.Name}}' minio | head -n 1)
  # --entrypoint sh, because the image's entrypoint is `mc` itself: passing
  # `sh -c ...` as the command made it run `mc sh`, which mc helpfully offered
  # to interpret as `mc share`.
  docker run --rm \
    --network "container:$NETWORK" \
    -e "MC_HOST_local=http://${AWS_ACCESS_KEY_ID:-minioadmin}:${AWS_SECRET_ACCESS_KEY:-minioadmin}@localhost:9000" \
    -v "$(pwd)/$SRC:/backup:ro" \
    --entrypoint sh \
    minio/mc:latest -c "mc mb --ignore-existing local/$S3_BUCKET && mc mirror --overwrite /backup/bucket local/$S3_BUCKET"
elif [ -f "$SRC/models.tar.gz" ]; then
  echo "  local disk: unpacking into the analyst-models volume"
  docker compose stop backend worker >/dev/null 2>&1 || true
  docker run --rm \
    -v "$(docker volume ls -q --filter name=analyst-models | head -n 1):/models" \
    -v "$(pwd)/$SRC:/backup:ro" \
    alpine:3 tar -xzf /backup/models.tar.gz -C /models
  # Both, not just the API: the worker was stopped too, and a stack whose
  # worker never came back looks like a queue that accepts and never runs.
  docker compose start backend worker >/dev/null 2>&1 || true
else
  echo "Backup has neither bucket/ nor models.tar.gz" >&2
  exit 1
fi

echo
echo "Restored. The schema is migrated to head by the API on its next start;"
echo "check GET /readyz, then run:  python ops/verify_restore.py"
[ -f "$SRC/manifest.json" ] && cat "$SRC/manifest.json"
