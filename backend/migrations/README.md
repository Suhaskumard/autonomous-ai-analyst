# Migrations

The schema is Alembic's from Phase 7 on. Before it, `db.init_db()` called
`SQLModel.metadata.create_all` and then patched in any column a later phase had
added. That worked for exactly one shape of change — an added, defaulted column
on SQLite — and silently did nothing for a rename, a type change, a new
constraint, or a backfill. With a worker and one or more API processes writing
to the same Postgres, "silently did nothing" is not a failure mode worth
keeping.

## Everyday use

```bash
cd backend
alembic revision --autogenerate -m "what changed"   # then read what it wrote
alembic upgrade head
alembic downgrade -1                                # if it was wrong
alembic current                                     # what this database is at
```

`DATABASE_URL` from the environment decides which database is migrated;
`alembic.ini` deliberately carries no URL, so the CLI and the application can
never disagree about that.

**Autogenerate is a first draft, not an answer.** It compares tables and
columns; it does not know that a renamed column is a rename rather than a drop
plus an add, and it will not write the backfill that makes the new column
correct. Read every generated script before committing it.

## What runs at startup

`db.init_db()` calls `utils.migrations.upgrade_to_head`, so both the API and
the worker bring the schema to head when they start, and whichever gets there
second finds nothing to do (on Postgres, an advisory lock makes that a wait
rather than a race). An empty database gets the whole chain applied normally.

A database created **before** Phase 7 has no `alembic_version` row and is
adopted rather than migrated: missing tables are created, missing columns are
added, and the result is stamped at `0001_baseline`. Both steps are needed,
because the old startup only ever created what the phase running at the time
knew about — the database in this repo has two of the six tables and none of
Phase 6's columns. Stamping that as complete would have declared a schema
correct while `users` did not exist.

One caveat worth knowing before it surprises someone: a column backfilled into
an existing table is nullable where the baseline says NOT NULL, because SQLite
cannot add a NOT NULL column to a table that already has rows. An adopted
database is compatible, not identical. To get one that matches the baseline
exactly, load a dump into a fresh database and let the chain run from empty.

Migrations at startup are the right trade for a two-process deployment: an
operator who has to remember a manual step before every deploy eventually
forgets, and a worker that starts before the schema exists writes job rows into
a table that is not there. If this ever grows to many replicas, move the
upgrade into a deploy job and leave the processes reading a schema they did not
create.

## Conventions

- `render_as_batch` is on, so a script that alters a table works on SQLite as
  well as Postgres. SQLite cannot `ALTER COLUMN`; batch mode rewrites the table.
- Timestamps are naive columns holding UTC. `utils.helpers.now_utc` writes them
  and `db._create_engine` pins the Postgres session timezone to UTC so a read
  gives back what was written.
- `tests/test_production_path.py` fails if the models and the migrations have
  drifted apart, which is what stops the next schema change from arriving
  without a script.
