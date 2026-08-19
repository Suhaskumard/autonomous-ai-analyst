# Hardening the edge

Phase 6 established *who* a caller is. This is what they may do, how much of it
they may do, and what is remembered afterwards.

**Skip all of it while every account belongs to someone you know.** Everything
here defaults to off, and that is not timidity: a per-account rate limit on a
single-operator install is a per-machine limit wearing a disguise, and the
containerised sandbox needs an image and a runtime that a laptop install has no
reason to have. Turning these on before there is an untrusted user costs
availability and buys nothing.

The moment any of it stops being true — an account you did not create, a
deployment more than one person can reach — work down this page in order.

---

## The order

| # | Turn on | Setting | Proven by |
|---|---------|---------|-----------|
| 1 | Per-account ceilings | `PRINCIPAL_*` | `tests/test_abuse.py` |
| 2 | Key rotation | nothing; the endpoints are always there | `POST /api/account/keys` |
| 3 | The audit log | nothing; it records once `AUTH_ENABLED` is on | `GET /api/admin/audit` |
| 4 | Secrets from files | `*_FILE` | `docker compose config` |
| 5 | The sandbox boundary | `SANDBOX_BACKEND=container` | `ops/verify_sandbox.py` |
| 6 | Container hardening | `docker-compose.hardened.yml` | `/readyz` |

One at a time, with `/readyz` in view. Steps 5 and 6 are the ones that can take
the application down if a flag is wrong, and they are last for that reason.

---

## 1. Ceilings, per account

The Phase 5 guards are per *conversation*: a message rate and a token budget,
both reset by starting a new one. That is the right guard for an operator
protecting themselves from a runaway script and the wrong one for an account
that might be hostile — a client in a loop opens a new conversation each time
and has no ceiling at all.

These are per account, and each is measured against a table that already exists.
`llm_usage` records every paid call with its estimated cost; `runs` records every
stored dataset; `jobs` records what is queued right now. Nothing new is tracked.
What was missing was somebody consulting it *before* the next call.

```bash
PRINCIPAL_MAX_LLM_CALLS_PER_HOUR=60     # paid endpoints: /api/chat, /api/report
PRINCIPAL_DAILY_SPEND_LIMIT_USD=5.00    # estimated, from the usage table
PRINCIPAL_MAX_RUNS=25                   # stored datasets
PRINCIPAL_STORAGE_QUOTA_MB=2048         # what those datasets occupy on disk
PRINCIPAL_MAX_CONCURRENT_JOBS=3         # queued or training at once
```

Every one defaults to `0`, meaning off.

Three properties worth knowing before you pick numbers:

- **The spend limit does not refund.** It is checked before a call and the cost
  is recorded after, so a single call may cross the line rather than being
  stopped exactly on it. Enforcing to the cent would mean pricing a response
  before generating it, which nobody can do.
- **The spend figure is an estimate**, from published list prices, not an
  invoice. A limit built on it inherits that. `/api/usage` says so in its
  response and so does this sentence.
- **`PRINCIPAL_MAX_RUNS` counts *stored* runs, so uploads still training do not
  count yet.** An account at 0 of 1 can start three uploads in the second before
  the first one finishes and end up over the ceiling, because each was checked
  against a registry none of them had reached. That is not a hole in the
  storage limit, it is the reason `PRINCIPAL_MAX_CONCURRENT_JOBS` exists —
  bounding the burst is the concurrency limit's job. **Set them together.** A
  deployment with `PRINCIPAL_MAX_RUNS` and no concurrency ceiling has a limit
  that can be walked past by anyone in a hurry.
- **The disk and worker limits are the ones that stop a *cheap* attack.**
  Uploading is free. An account that never touches a paid endpoint can still
  fill the disk, and an account under both of those ceilings can still occupy
  every worker with training runs. `PRINCIPAL_MAX_CONCURRENT_JOBS` is a ceiling
  rather than fair scheduling, which would be the better answer and a much
  larger change: fair queueing means RQ has to know about principals, and it
  does not.

A refusal is a `429` (or `413` for storage), and every refusal is written to the
audit log — a client hammering a ceiling is exactly the pattern worth finding
later, and "the system was slow that afternoon" is much easier to explain with
those rows in hand.

Callers can see where they stand at `GET /api/usage`, under `quota`.

---

## 2. Rotating a key without an outage

Phase 6 put one key hash on the user row. With one credential per account,
rotation *is* an outage the length of the rollout: the new key does not work
until the old one has stopped. Keys now live in their own table, several may be
active at once, and rotation is three steps with no gap.

```bash
# 1. issue. Both keys work from this moment.
curl -X POST http://localhost:8000/api/account/keys \
     -H "Authorization: Bearer $OLD_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"label": "rotation 2026-08", "ttl_days": 90}'

# 2. roll the new key out to every caller, and let them settle.

# 3. revoke the old one. Its blast radius is exactly the callers still using it.
curl -X DELETE http://localhost:8000/api/account/keys/$OLD_KEY_ID \
     -H "Authorization: Bearer $NEW_KEY"
```

`GET /api/account/keys` lists them with `last_used_at`, which is how you tell
whether step 2 finished before you do step 3.

Decisions in here that the endpoints do not explain:

- **Rotation is the account's own to do**, not an admin's. The person who has to
  update every caller is the one who should be able to mint the replacement.
  Requiring an admin for a routine rotation is how rotations stop happening.
- **The last active key cannot be revoked** (`409`). An account with no working
  credential cannot issue itself a new one, and recovering from that needs an
  admin and a conversation. Deactivate the account if that was the intent.
- **Revocation is a timestamp, not a delete.** "Which key made this request, and
  when did we stop trusting it" is a question the audit log has to be able to
  answer after the fact.
- **Keys issued before this phase still work.** They exist only on the user row,
  and authentication falls back to it. The migration copies each one into the
  key table so it can also be listed, labelled, and revoked.
- `MAX_ACTIVE_API_KEYS=5` keeps "rotate" from quietly becoming "accumulate", and
  `API_KEY_DEFAULT_TTL_DAYS=0` means issued keys do not expire unless you say so.

### Blast radius

| Action | What stops working | What does not |
|---|---|---|
| Revoke one key | callers holding that key | the account, its data, its other keys |
| Deactivate an account | every key on it | its data, and the audit log of what it did |
| Rotate `ARTIFACT_SIGNING_KEY` | loading every existing model bundle | uploading and retraining |

The third row is the one that surprises people. Bundles are verified against the
signing key before `joblib.load` sees them, so changing that key makes every
already-trained model unloadable. It is not part of key rotation and should not
be rotated on the same schedule.

---

## 3. The audit log

Privileged and destructive actions only: account creation and deactivation, key
issuance and revocation, dataset deletion, account erasure, retention purges,
and refused quota. Reading a run is not audited — a log that records everything
is a second copy of the access log that nobody reads.

```bash
curl -H "Authorization: Bearer $ADMIN_KEY" \
     'http://localhost:8000/api/admin/audit?limit=100&action=run.deleted'
```

An account can see its own actions at `GET /api/account/audit`. Which admin
deactivated whom is not an account holder's business, so that stays admin-only.

Three properties, each a decision:

- **Append-only.** `db` has `append_audit` and `list_audit` and no update or
  delete partner. Nothing in this application edits history.
- **Outside retention.** `lifecycle.purge_expired` deletes runs, artifacts and
  conversations and does not touch this table. The record that a dataset was
  deleted has to outlive the dataset.
- **No payload.** Entries carry ids and counts, never dataset contents, column
  names, or question text. The log is read by more people than the data is.

With `AUTH_ENABLED=false` every entry is attributed to the same implicit local
owner, so the log records a sequence of events but not *who*. `GET
/api/admin/audit` says so in its `note` rather than implying otherwise.

---

## 4. Secrets out of the environment

Environment variables are readable by anything that can list a process, take a
core dump, or run `docker inspect`, and they end up in shell history and CI
logs. Docker secrets, Kubernetes secret volumes, and systemd credentials all
present a secret as a *file* instead.

So each of these accepts a `_FILE` variant naming a path:

```bash
GEMINI_API_KEY_FILE=/run/secrets/gemini
ARTIFACT_SIGNING_KEY_FILE=/run/secrets/signing
AUTH_BOOTSTRAP_TOKEN_FILE=/run/secrets/bootstrap
DATABASE_URL_FILE=/run/secrets/database-url
```

That is deliberately not an integration with any particular secret manager.
Every one of them can present a file, and a file is a contract this application
can support without learning about four vendors' SDKs.

The direct variable still works and still wins. Setting both logs a warning:
the two disagreeing silently is how the wrong credential gets used. A `_FILE`
that cannot be read or is empty is fatal at startup, because falling back to
"no key" would look like a configuration choice.

With Docker Compose:

```yaml
services:
  backend:
    environment:
      GEMINI_API_KEY_FILE: /run/secrets/gemini
    secrets: [gemini]
secrets:
  gemini:
    file: ./secrets/gemini.txt   # or external: true
```

**Still outstanding from Phase 0 and not fixed by any of this:** the Gemini key
that was committed to git history and baked into every backend image built
before Phase 7 added `backend/.dockerignore`. Moving to a file does nothing for
a key that has already leaked. Rotate it at the provider.

---

## 5. The sandbox, as an actual boundary

`analyst/runner.py` has said plainly since Phase 5 what it is: an import
denylist and a PEP 578 audit hook, which stop generated code that *misbehaves*
and would not stop generated code that *attacks*. Both live inside the
interpreter they guard. Anything with arbitrary Python eventually gets
underneath them.

```bash
docker build -f ops/Dockerfile.sandbox -t analyst-sandbox:latest ./backend
python ops/verify_sandbox.py --build
```

Then:

```bash
SANDBOX_BACKEND=container
SANDBOX_IMAGE=analyst-sandbox:latest
SANDBOX_SECCOMP_PROFILE=ops/seccomp-analyst.json   # optional; see below
```

The same runner, unchanged, now executes inside a container with **no network
namespace**, a **read-only root**, **every capability dropped**,
`no-new-privileges`, a pids limit, a memory limit that swap cannot evade, and
**exactly one dataset mounted read-only**. That last one closes a gap the
interpreter guards never addressed at all: under the subprocess backend an
execution can read the whole `models/` tree, including other accounts'
snapshots.

**`ops/verify_sandbox.py` is not optional.** It makes nine assertions against a
real container, and two of them — no network, no writes to `/` — are written to
fail if the isolation is absent rather than to pass if it is present. Every part
of this can be true in the config and false on the machine: the image may not be
built, the profile may deny a syscall numpy needs, the mount may resolve to
nothing.

### What it costs

- **About a second per execution**, for the container start.
  `SANDBOX_STARTUP_GRACE_SECONDS` is added to the timeout so the analysis still
  gets the time it was promised.
- **Access to a container runtime.** When the API is itself a container, that
  means a mounted socket. `/var/run/docker.sock` **is root on the host**: it
  exchanges "generated code escapes into the API container" for "the API is
  compromised and owns the machine". That is a good trade when the analyst is
  the exposed surface and the API is not, and it is not one to make blind.
  Prefer a **rootless Podman socket** — same flags, same image, no root:

  ```bash
  SANDBOX_RUNTIME=podman
  # mount ${XDG_RUNTIME_DIR}/podman/podman.sock at /var/run/docker.sock
  ```

- **`SANDBOX_HOST_MODELS_DIR`**, in that same topology, and this is the step that
  gets missed. Bind mounts are resolved by the *daemon*, against the host
  filesystem, so the path the API sees for a snapshot is not the path the daemon
  must be given. Set it to the host directory behind the `analyst-models`
  volume. Wrong or unset, the mount silently produces an empty file, which
  surfaces as "the dataset has no rows" and looks nothing like a mount problem.

### Refusing rather than degrading

Elsewhere this application degrades: an unreachable Redis falls back to inline
execution, S3 to local disk. Those are availability features, and a slow answer
beats no answer.

This is not one. If `SANDBOX_BACKEND=container` and the runtime is missing, or
the seccomp profile named cannot be read, **every execution fails** and none
falls back to the subprocess backend. An operator who asked for a boundary and
quietly did not get one is the single failure mode here worth an outage.
`/readyz` reports `checks.sandbox.ok: false` for exactly that case.

### The seccomp profile

`ops/seccomp-analyst.json` is an **allowlist** — 136 syscalls, `defaultAction:
SCMP_ACT_ERRNO` — and that shape is the whole point. A custom profile *replaces*
the runtime's default rather than adding to it, so a profile written as "deny
these dangerous calls" silently permits everything the default blocked and
nobody listed. Denied on purpose and named in the file: `socket`, `ptrace`,
`mount`, `unshare`, `setns`, `bpf`, `init_module`, `keyctl`, `userfaultfd`, and
the rest.

Two caveats:

1. **It is optional.** Leave `SANDBOX_SECCOMP_PROFILE` unset and the runtime's
   default profile applies, which is itself an allowlist and already good. The
   shipped profile is tighter, not load-bearing.
2. **It has not been exercised against a running daemon in this repository** —
   the machine it was written on had no daemon available. `ops/verify_sandbox.py
   --seccomp ops/seccomp-analyst.json` is the check that turns it from plausible
   into known, and a syscall missing from the list shows up there as an import
   failure in numpy or scipy rather than as a subtly wrong answer. Run it before
   trusting it.

It is scoped to the sandbox and **must not** be applied to the API or worker
containers, whose syscall surface — a web server, a Postgres driver, xgboost —
is far wider than this.

---

## 6. Container hardening

```bash
docker compose -f docker-compose.yml -f docker-compose.hardened.yml \
               --profile queue up -d
```

Read-only roots, all capabilities dropped, `no-new-privileges`, memory and pids
limits, and one tmpfs per service for the paths that genuinely need writing.

This is the smaller half of the sandbox story. It bounds the blast radius **of
the API and worker processes**; it does not isolate generated code, which runs
wherever `SANDBOX_BACKEND` says. Applied without step 5, every protection in it
is one the analysis code *shares* rather than one it is subject to.

---

## What Phase 8 does not do

- **No fair scheduling.** `PRINCIPAL_MAX_CONCURRENT_JOBS` refuses the
  fifty-first upload; it does not give each account a turn.
- **The rate limiter queries the database per guarded call**, so two API
  processes share one count without needing Redis. That trade is right while the
  guarded endpoints take seconds anyway, and it is the wrong shape for a limit
  on something cheap.
- **No IP-level or unauthenticated rate limiting.** Every ceiling here is
  per-principal, so they do nothing about a flood at endpoints that need no key.
  That belongs at the ingress.
- **The audit log is append-only by convention**, enforced by the absence of any
  update or delete helper. It is not cryptographically chained; someone with
  direct database access can still edit it.
- **The sandbox boundary is per-execution isolation, not multi-tenancy.** Two
  executions cannot see each other because neither outlives its container, but
  they share a kernel, and a kernel exploit is a kernel exploit.
