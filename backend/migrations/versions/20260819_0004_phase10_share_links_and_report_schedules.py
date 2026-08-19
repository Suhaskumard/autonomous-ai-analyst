"""Phase 10: share links and report schedules.

Two tables, neither touching an existing one.

`share_links` is the credential behind a read-only, expiring link onto one
run's dashboard result — see sharing.py for what it does and does not grant.
`expires_at` is NOT NULL: unlike a run's own retention, which defaults to
"keep forever", a link with no expiry would be a permanent unauthenticated
door, so there is no version of this table that allows one.

`report_schedules` is a standing instruction to regenerate a run's report and
email it on a cadence. Nothing in this migration or the application starts a
scheduler — `next_run_at` is a due date an external cron reads via
`POST /api/report-schedules/run-due`, the same shape Phase 6's retention purge
already uses.

Revision ID: 0004_phase10
Revises: 0003_monitoring
Created: 2026-08-19

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "0004_phase10"
down_revision: str | None = "0003_monitoring"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "share_links",
        sa.Column("token", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("run_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("owner_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("token"),
    )
    with op.batch_alter_table("share_links", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_share_links_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_share_links_expires_at"), ["expires_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_share_links_owner_id"), ["owner_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_share_links_revoked_at"), ["revoked_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_share_links_run_key"), ["run_key"], unique=False)

    op.create_table(
        "report_schedules",
        sa.Column("schedule_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("run_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("owner_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("recipients", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("interval", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("next_run_at", sa.DateTime(), nullable=False),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("last_status", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("schedule_id"),
    )
    with op.batch_alter_table("report_schedules", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_report_schedules_active"), ["active"], unique=False)
        batch_op.create_index(batch_op.f("ix_report_schedules_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_report_schedules_next_run_at"), ["next_run_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_report_schedules_owner_id"), ["owner_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_report_schedules_run_key"), ["run_key"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("report_schedules", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_report_schedules_run_key"))
        batch_op.drop_index(batch_op.f("ix_report_schedules_owner_id"))
        batch_op.drop_index(batch_op.f("ix_report_schedules_next_run_at"))
        batch_op.drop_index(batch_op.f("ix_report_schedules_created_at"))
        batch_op.drop_index(batch_op.f("ix_report_schedules_active"))
    op.drop_table("report_schedules")

    with op.batch_alter_table("share_links", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_share_links_run_key"))
        batch_op.drop_index(batch_op.f("ix_share_links_revoked_at"))
        batch_op.drop_index(batch_op.f("ix_share_links_owner_id"))
        batch_op.drop_index(batch_op.f("ix_share_links_expires_at"))
        batch_op.drop_index(batch_op.f("ix_share_links_created_at"))
    op.drop_table("share_links")
