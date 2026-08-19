"""Phase 8: api_keys and audit_log.

Two tables and one backfill.

`api_keys` moves the credential off the user row, where there was room for
exactly one, so an account can hold several at once and rotation stops being an
outage. The backfill copies each existing user's key hash into the new table:
without it, keys issued before this migration would still authenticate (auth.py
falls back to the user row) but could not be listed, labelled, or revoked —
which is most of the point.

`audit_log` is append-only and outside the retention policy. Nothing in the
application updates or deletes a row in it.

Revision ID: 0002_keys_audit
Revises: 0001_baseline
Created: 2026-08-18

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "0002_keys_audit"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("key_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("user_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("key_hash", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("label", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("key_id"),
    )
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_api_keys_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_api_keys_expires_at"), ["expires_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_api_keys_key_hash"), ["key_hash"], unique=False)
        batch_op.create_index(batch_op.f("ix_api_keys_revoked_at"), ["revoked_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_api_keys_user_id"), ["user_id"], unique=False)

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("actor_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("actor_email", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("action", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("target_type", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("target_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("request_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("audit_log", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_audit_log_action"), ["action"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_log_actor_id"), ["actor_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_log_at"), ["at"], unique=False)
        batch_op.create_index(batch_op.f("ix_audit_log_target_id"), ["target_id"], unique=False)

    # Backfill: every existing account's key becomes a manageable key row.
    # `legacy_` + user_id as the id is deterministic, so re-running this on a
    # database that already has the row is a conflict rather than a duplicate.
    # created_at copies the user's, which is when that key really was issued.
    op.execute(
        """
        INSERT INTO api_keys (key_id, user_id, key_hash, label, created_at, expires_at, revoked_at, last_used_at)
        SELECT 'legacy_' || user_id, user_id, api_key_hash, 'original', created_at, NULL, NULL, last_seen_at
        FROM users
        WHERE api_key_hash IS NOT NULL
          AND user_id NOT IN (SELECT user_id FROM api_keys)
        """
    )


def downgrade() -> None:
    with op.batch_alter_table("audit_log", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_audit_log_target_id"))
        batch_op.drop_index(batch_op.f("ix_audit_log_at"))
        batch_op.drop_index(batch_op.f("ix_audit_log_actor_id"))
        batch_op.drop_index(batch_op.f("ix_audit_log_action"))

    op.drop_table("audit_log")
    with op.batch_alter_table("api_keys", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_api_keys_user_id"))
        batch_op.drop_index(batch_op.f("ix_api_keys_revoked_at"))
        batch_op.drop_index(batch_op.f("ix_api_keys_key_hash"))
        batch_op.drop_index(batch_op.f("ix_api_keys_expires_at"))
        batch_op.drop_index(batch_op.f("ix_api_keys_created_at"))

    op.drop_table("api_keys")
