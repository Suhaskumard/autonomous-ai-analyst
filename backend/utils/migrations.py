"""Running Alembic from inside the application.

Two processes start this application — the API and the worker — and either may
be first. So migrations cannot be a step an operator remembers to run before
the deploy: they run at startup, under a lock, and the second process to arrive
finds the schema already at head and does nothing.

The awkward case this handles is the upgrade *into* Alembic. A database written
before Phase 7 has no `alembic_version` row, so an unconditional `upgrade head`
would try to create tables that already exist — and it will not have the full
baseline either, because the old `create_all` + patch-in-missing-columns startup
only ever ran as far as whichever phase last touched that machine. The real one
in this repo has two of the six tables and none of Phase 6's columns.

So adoption is a three-step: bring the database up to what `create_all` would
have produced, stamp it at the revision it actually satisfies, then upgrade
normally. The additive column patch is the last surviving piece of the old
helper and runs only on this path, once, for a database that predates Alembic.

Two things about that middle step are easy to get wrong, and the first version
of this module got both:

* **"What `create_all` would have produced" means at the baseline, not today.**
  Building it from live `SQLModel.metadata` creates every table the *current*
  models declare, including ones added by later revisions — and then stamping
  the baseline makes the next `create_table` migration fail on a table that is
  already there. The target schema is therefore read from the baseline
  migration itself, by replaying it on a scratch database, so it stays correct
  as revisions are added without anyone having to remember this.
* **The stamp is not always the baseline.** A database can predate Alembic and
  still have a later revision's tables — anything built by the old `create_all`
  after that revision's models existed. Stamping it at the baseline would
  replay migrations over tables that are already there. So the chain is walked
  forward and the database is stamped at the newest revision it already
  satisfies.

Both are decided by reading the migration scripts, never by a hardcoded list of
tables: a list is a thing that goes stale silently, one phase later.
"""

import logging
from functools import lru_cache
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = BACKEND_DIR / "migrations"

#: The revision that corresponds to the pre-Alembic schema. A database that
#: already has tables but no version row is brought up to this and stamped at
#: it, rather than having it run as a migration over tables that exist.
BASELINE_REVISION = "0001_baseline"

#: A table that only ever existed in a database this application created. Used
#: to tell "pre-Alembic install" apart from "empty database".
_SENTINEL_TABLE = "runs"

#: Advisory lock id for the migration itself. Arbitrary but fixed: it only has
#: to be the same number in every process that migrates this database.
_MIGRATION_LOCK_ID = 8_675_309


def alembic_config(connection=None):
    """An Alembic config wired to this package rather than to a CLI cwd.

    The script location is resolved from this file, so `alembic upgrade head`
    behaves the same whether it is run by the API at startup, by the worker, or
    by hand from any directory.
    """
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    # env.py prefers this over creating a second engine, so a migration run
    # inside a test uses that test's temporary database.
    if connection is not None:
        config.attributes["connection"] = connection
    return config


def current_revision(connection) -> str | None:
    from alembic.runtime.migration import MigrationContext

    return MigrationContext.configure(connection).get_current_revision()


def head_revision() -> str:
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def _take_migration_lock(connection) -> None:
    """Serialise concurrent migrations, where the database can do that.

    The API and the worker start together and both migrate. On Postgres, DDL is
    transactional, so without this one of them loses the race and exits on a
    duplicate-table error — survivable, since the container restarts and finds
    the schema already at head, but it looks like a crash loop for a few
    seconds and it is avoidable. The lock is held for the transaction and
    released with it, including if this process dies mid-migration.

    SQLite has no advisory lock and needs none: its writer lock is the file, and
    nothing that runs on SQLite here has two processes writing anyway.
    """
    if connection.dialect.name != "postgresql":
        return
    connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _MIGRATION_LOCK_ID})


def _legacy_column_default(column) -> str | None:
    """A literal default for a column being backfilled into a legacy table.

    Only the ones whose right value is knowable without looking at the data:
    every pre-Phase-6 row belongs to the single local owner, and a token count
    that was never recorded is zero. Anything else is added nullable and left
    for the application to fill.
    """
    from db import LOCAL_OWNER_ID

    if column.name == "owner_id":
        return f"'{LOCAL_OWNER_ID}'"
    if column.name in {"input_tokens", "output_tokens"}:
        return "0"
    return None


@lru_cache(maxsize=1)
def _schema_by_revision() -> list[tuple[str, dict[str, set[str]]]]:
    """[(revision, {table: {columns}})] after each revision, oldest first.

    Built by replaying the whole chain on a throwaway in-memory SQLite database
    and inspecting it between steps. That is more work than a hardcoded table
    list and it is the reason this keeps working: the migration scripts are the
    definition of every schema this application has ever had, so anything
    derived from them cannot disagree with them. A list maintained by hand
    would be right today and quietly wrong one phase from now — which is
    exactly how the bug this replaced got in.

    SQLite is fine as the scratch dialect even when the real database is
    Postgres: what is wanted here is the *names*, and those do not vary.
    """
    from alembic.script import ScriptDirectory
    from sqlalchemy import create_engine

    script = ScriptDirectory.from_config(alembic_config())
    # `walk_revisions` yields newest first; the chain has to be replayed the
    # other way round.
    revisions = [revision.revision for revision in reversed(list(script.walk_revisions()))]

    schema: list[tuple[str, dict[str, set[str]]]] = []
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            config = alembic_config(connection)
            for revision in revisions:
                command.upgrade(config, revision)
                inspector = inspect(connection)
                schema.append(
                    (
                        revision,
                        {
                            table: {column["name"] for column in inspector.get_columns(table)}
                            for table in inspector.get_table_names()
                            if table != "alembic_version"
                        },
                    )
                )
    finally:
        engine.dispose()
    return schema


def _satisfies(connection, expected: dict[str, set[str]]) -> bool:
    """Whether this database already has every table and column `expected` names.

    Deliberately a subset test, not an equality one. A database being adopted
    may carry columns from a later revision, or columns this application has
    never declared; neither means it is missing the revision being checked.
    """
    inspector = inspect(connection)
    present = set(inspector.get_table_names())
    for table, columns in expected.items():
        if table not in present:
            return False
        if not columns <= {column["name"] for column in inspector.get_columns(table)}:
            return False
    return True


def _revision_already_satisfied(connection) -> str:
    """The newest revision whose schema this database already has.

    Walked forward and stopped at the first gap rather than scanning for the
    best match: revisions are a chain, and a database that has revision 3's
    tables but not revision 2's is not "at 3", it is inconsistent, and stamping
    it at 3 would skip a migration it genuinely needs.
    """
    stamp = BASELINE_REVISION
    for revision, expected in _schema_by_revision():
        if not _satisfies(connection, expected):
            break
        stamp = revision
    return stamp


def _adopt_pre_alembic_schema(connection) -> str:
    """Bring a pre-Phase-7 database to the baseline, then stamp what it has.

    `create_all` adds the tables that phase never created — a Phase 2 database
    has `jobs` and `runs` and nothing else. The column loop is what the retired
    `_add_missing_columns` did, and it is still needed here for exactly the
    reason it existed: Phase 6 added `owner_id` to tables Phase 2 created, and
    there is no Alembic revision that adds it, because the baseline *includes*
    it.

    Both halves are scoped to the **baseline** schema rather than to today's
    models. Creating a later revision's tables here would leave nothing for that
    revision's `create_table` to do but fail — and, worse, would skip the data
    migration that goes with it. `0002` copies every existing user's key hash
    into `api_keys`; a database handed that table pre-made by `create_all` gets
    an empty one, and every credential in circulation silently stops being
    listable or revocable.

    A column backfilled this way is nullable where the baseline says NOT NULL —
    SQLite cannot add a NOT NULL column to a populated table without a default
    for it. The result is compatible rather than identical; a database that has
    to match the baseline exactly needs a dump and a reload into a fresh one.

    Returns the revision it stamped, which is not always the baseline: see
    `_revision_already_satisfied`.
    """
    from sqlmodel import SQLModel

    inspector = inspect(connection)
    existing = set(inspector.get_table_names())
    baseline = dict(_schema_by_revision())[BASELINE_REVISION]

    missing = [
        table for table in SQLModel.metadata.sorted_tables if table.name in baseline and table.name not in existing
    ]
    if missing:
        SQLModel.metadata.create_all(connection, tables=missing, checkfirst=True)
        logger.info(
            "Created baseline tables missing from a pre-Alembic database: %s",
            ", ".join(sorted(table.name for table in missing)),
        )

    for table in SQLModel.metadata.sorted_tables:
        if table.name not in existing or table.name not in baseline:
            continue  # just created, or introduced after the baseline
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present or column.name not in baseline[table.name]:
                continue
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {column.type.compile(connection.dialect)}'
            default = _legacy_column_default(column)
            if default is not None:
                ddl += f" DEFAULT {default}"
            logger.info("Backfilling %s.%s into a pre-Alembic database", table.name, column.name)
            connection.execute(text(ddl))

    stamp = _revision_already_satisfied(connection)
    command.stamp(alembic_config(connection), stamp)
    return stamp


def upgrade_to_head(engine) -> str | None:
    """Migrate the schema to head. Returns the revision now in place."""
    with engine.begin() as connection:
        _take_migration_lock(connection)
        version = current_revision(connection)
        if version is None and _SENTINEL_TABLE in set(inspect(connection).get_table_names()):
            stamped = _adopt_pre_alembic_schema(connection)
            logger.info("Adopted a pre-Alembic schema and stamped it at %s", stamped)

        config = alembic_config(connection)
        command.upgrade(config, "head")
        applied = current_revision(connection)

    if applied != version:
        logger.info("Schema migrated", extra={"from": version, "to": applied})
    return applied


def pending_revisions(engine) -> list[str]:
    """Revisions this database has not applied yet — what /readyz reports."""
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(alembic_config())
    with engine.connect() as connection:
        version = current_revision(connection)
    return [revision.revision for revision in script.iterate_revisions("head", version) if revision.revision != version]


def schema_status(engine) -> dict:
    """Whether the schema this process expects is the schema it is talking to."""
    try:
        with engine.connect() as connection:
            version = current_revision(connection)
        head = head_revision()
        return {
            "ok": version == head,
            "revision": version,
            "head": head,
        }
    except Exception as exc:  # pragma: no cover - unreachable database
        return {"ok": False, "detail": str(exc)}
