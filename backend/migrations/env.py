"""Alembic environment.

Two entry points share this file: `alembic` on the command line, and the
application calling `utils.migrations.upgrade_to_head` at startup. The
difference is whether a live connection was handed over in
`config.attributes`; everything else — the URL, the metadata, the render
options — is the same in both, so a migration cannot behave one way in a
deployment and another way in a test.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sqlmodel import SQLModel  # noqa: E402

# Importing db is what populates SQLModel.metadata: every table is declared as
# a side effect of the model classes. Without this, autogenerate would compare
# the database against nothing and cheerfully propose dropping every table.
import db  # noqa: E402,F401  (imported for its table definitions)
from config import settings  # noqa: E402
from db import normalise_database_url  # noqa: E402

config = context.config

# Only when run from the CLI. In-process runs keep the application's logging
# configuration rather than replacing it mid-startup.
if config.config_file_name is not None and context.config.attributes.get("connection") is None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def _database_url() -> str:
    # The environment wins over alembic.ini, which carries no real URL: the
    # deployment's DATABASE_URL is the single source of truth for where the
    # schema lives, and a stale ini value would migrate the wrong database.
    return normalise_database_url(settings.database_url)


def _configure(**kwargs) -> None:
    context.configure(
        target_metadata=target_metadata,
        # SQLite cannot ALTER most things in place; batch mode rewrites the
        # table instead, so one migration script works on both backends.
        render_as_batch=True,
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout — `alembic upgrade head --sql`, for a reviewed change."""
    _configure(url=_database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _configure(connection=connection)
        context.run_migrations()
        return

    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as conn:
        _configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
