#!/usr/bin/env sh
# Take a restorable backup of a running stack.
#
#   ops/backup.sh [destination]
#
# Two things have to come out together or the backup is worthless: the database
# (which knows a run exists, who owns it, and when it expires) and the artifact
# bytes (the model, its signature, the metadata, the snapshot). A database
# without artifacts lists runs that 404; artifacts without a database are
# unreachable while authentication is on, by design — `auth.require_owned_run`
# refuses a file with no registry row.
#
# What it does not back up: Redis. A queue holds jobs in flight, not results,
# and a job lost to a restore is a re-upload rather than a lost model.
#
# Restore with ops/restore.sh, then prove it with ops/verify_restore.py. A
# backup nobody has restored is a hypothesis.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_ROOT"

# Git Bash on Windows rewrites arguments that look like absolute POSIX paths
# into Windows paths before the command sees them, so the *container* path
# /backup became C:/Program Files/Git/backup and mc mirrored the bucket to a
# directory inside the Git installation. The backup looked like it worked: the
# transfer summary was green and the files were simply somewhere else.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST=${1:-backups/$STAMP}
mkdir -p "$DEST"

POSTGRES_USER=${POSTGRES_USER:-analyst}
POSTGRES_DB=${POSTGRES_DB:-analyst}
S3_BUCKET=${S3_BUCKET:-analyst-artifacts}

# DATABASE_URL is what the application actually reads, so it decides what gets
# dumped. POSTGRES_DB only says which database the *server* created on first
# boot, and the two are free to disagree — when they did, this script dumped an
# empty database, reported success, and produced an 800-byte "backup".
parse_database_url() {
  case "${DATABASE_URL:-}" in
    postgres*)
      _rest=${DATABASE_URL#*://}
      _creds=${_rest%%@*}
      _db=${_rest##*/}
      _db=${_db%%\?*}
      [ -n "$_db" ] && POSTGRES_DB=$_db
      case "$_creds" in
        *:*|*) [ -n "${_creds%%:*}" ] && POSTGRES_USER=${_creds%%:*} ;;
      esac
      ;;
  esac
}
parse_database_url

running() {
  docker compose ps --services --status running 2>/dev/null | grep -qx "$1"
}

echo "Backing up into $DEST"

# --- the database ------------------------------------------------------------
if running postgres; then
  echo "  postgres: pg_dump (custom format, restorable with pg_restore)"
  docker compose exec -T postgres \
    pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=custom --no-owner \
    > "$DEST/database.dump"
  DATABASE_KIND=postgres
else
  # A live SQLite file cannot be copied byte-for-byte: a checkpoint mid-copy
  # gives a torn database. The backup API takes a consistent snapshot of a
  # database that is being written to, which is exactly the situation here.
  echo "  sqlite: online backup via the sqlite3 backup API"
  docker compose exec -T backend python -c "
import sqlite3, sys
source = sqlite3.connect('/app/models/analyst.db')
target = sqlite3.connect('/tmp/analyst-backup.db')
with target:
    source.backup(target)
target.close()
sys.stderr.write('snapshot taken\n')
"
  docker compose exec -T backend sh -c 'cat /tmp/analyst-backup.db; rm -f /tmp/analyst-backup.db' \
    > "$DEST/analyst.db"
  DATABASE_KIND=sqlite
fi

# --- the artifacts -----------------------------------------------------------
if running minio; then
  echo "  minio: mirroring s3://$S3_BUCKET"
  # A throwaway mc container on the compose network, so this works whether or
  # not the host has mc installed and without publishing MinIO to the world.
  NETWORK=$(docker compose ps --format '{{.Name}}' minio | head -n 1)
  docker run --rm \
    --network "container:$NETWORK" \
    -e "MC_HOST_local=http://${AWS_ACCESS_KEY_ID:-minioadmin}:${AWS_SECRET_ACCESS_KEY:-minioadmin}@localhost:9000" \
    -v "$(pwd)/$DEST:/backup" \
    minio/mc:latest mirror --overwrite "local/$S3_BUCKET" "/backup/bucket"
  ARTIFACT_KIND=bucket
else
  echo "  local disk: archiving the analyst-models volume"
  # Read the volume through a throwaway container: the files belong to the
  # image's user and are not on the host filesystem in any portable place.
  docker run --rm \
    -v "$(docker volume ls -q --filter name=analyst-models | head -n 1):/models:ro" \
    -v "$(pwd)/$DEST:/backup" \
    alpine:3 tar -czf /backup/models.tar.gz -C /models .
  ARTIFACT_KIND=volume
fi

# --- what this is ------------------------------------------------------------
# Recorded rather than remembered: a restore six months from now needs to know
# which schema revision the dump came from and which signing key verifies the
# bundles in it. The key itself is deliberately not written here.
SCHEMA_REVISION=$(curl -fsS http://localhost:8000/readyz 2>/dev/null \
  | python -c "import json,sys; print(json.load(sys.stdin)['checks']['schema']['revision'])" 2>/dev/null \
  || echo "unknown")

cat > "$DEST/manifest.json" <<EOF
{
  "taken_at": "$STAMP",
  "database": "$DATABASE_KIND",
  "artifacts": "$ARTIFACT_KIND",
  "schema_revision": "$SCHEMA_REVISION",
  "bucket": "$S3_BUCKET",
  "artifact_signing_key_set": $( [ -n "${ARTIFACT_SIGNING_KEY:-}" ] && echo true || echo false ),
  "note": "Restoring needs the same ARTIFACT_SIGNING_KEY this stack ran with. Bundles signed with a key that is not in this file and not in your secret store cannot be verified, and will be refused rather than loaded."
}
EOF

echo "Done. $DEST"
ls -la "$DEST"
