"""Parent side of the sandbox: spawn, feed, time out, parse.

The README has promised server-side execution since before it existed. This
is the piece that makes it true — generated pandas runs here, against the
snapshot on this disk, and the numbers it returns can be checked by hand.

Everything about the child is deliberately one-shot: a fresh interpreter per
call, no state carried between requests, and a hard kill on the deadline. A
runaway loop costs one process, not the server.

There are two ways to start that child and this module owns the choice between
them. By default it is a subprocess of this one, guarded by the conventions in
`runner.py`. With `SANDBOX_BACKEND=container` it is the same runner inside a
container with no network namespace and a read-only root — see
`analyst/container.py` for what that buys and what it costs. Everything after
the spawn is identical, because the runner's contract is one JSON request on
stdin and one JSON response on stdout, and neither end of that cares which of
the two started it.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from analyst import container
from analyst.runner import RESULT_MARKER
from config import settings
from observability import metrics

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve().parent / "runner.py"

# A generous ceiling on the code itself; the model has no reason to emit more.
MAX_CODE_CHARS = 20_000

# The child gets what an interpreter needs to start and nothing else. Passing
# the whole environment would hand analysis code the Gemini API key; passing
# none of it breaks DLL resolution on Windows and locale on POSIX.
_ENV_PASSTHROUGH = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LANG", "LC_ALL", "TZ")


def _child_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in _ENV_PASSTHROUGH if name in os.environ}
    env["MPLBACKEND"] = "Agg"
    # matplotlib looks for a home directory to cache its font list in, and the
    # child is not given one. Point it somewhere disposable instead.
    env["MPLCONFIGDIR"] = str(_mpl_cache_dir())
    # Keeps a crashing child from writing .pyc files next to the runner.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _mpl_cache_dir() -> Path:
    """A shared scratch dir for matplotlib's font cache.

    Shared rather than per-call on purpose: rebuilding the font cache on every
    execution would add seconds to every question the analyst answers.
    """
    path = Path(tempfile.gettempdir()) / "analyst-mpl-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ExecutionResult:
    """What a sandbox run produced, in a shape the API can return verbatim."""

    ok: bool
    code: str
    stdout: str = ""
    stdout_truncated: bool = False
    error: str | None = None
    result: dict | None = None
    figures: list[dict] = field(default_factory=list)
    duration_ms: int = 0
    limits: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "code": self.code,
            "stdout": self.stdout,
            "stdout_truncated": self.stdout_truncated,
            "error": self.error,
            "result": self.result,
            "figures": self.figures,
            "duration_ms": self.duration_ms,
        }

    def summarise(self, limit: int = 4000) -> str:
        """A compact transcript for the model — what it would have seen in a REPL."""
        if not self.ok:
            return f"Execution failed: {self.error}"

        parts: list[str] = []
        if self.stdout.strip():
            parts.append(f"stdout:\n{self.stdout.strip()}")
        if self.result:
            kind = self.result.get("kind")
            if kind == "table":
                columns = ", ".join(self.result["columns"])
                rows = self.result["rows"][:20]
                rendered = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
                shape = self.result.get("shape", [len(self.result["rows"]), len(self.result["columns"])])
                parts.append(f"result table ({shape[0]} rows x {shape[1]} cols)\n{columns}\n{rendered}")
            elif kind == "scalar" or kind == "text":
                parts.append(f"result: {self.result['value']}")
        if self.figures:
            parts.append(f"({len(self.figures)} figure(s) rendered and shown to the user)")
        return ("\n\n".join(parts) or "The code ran and produced no output.")[:limit]


def run_code(code: str, snapshot_path: Path | str | None, timeout_seconds: int | None = None) -> ExecutionResult:
    """Execute `code` in a fresh sandboxed interpreter and return what it produced.

    A thin wrapper over `_run_code` so the outcome is counted on every path.
    There are six ways out of that function and five of them are failures; a
    metric recorded at each one is a metric that will be wrong the first time a
    sixth is added.
    """
    result = _run_code(code, snapshot_path, timeout_seconds)
    metrics.observe_sandbox("ok" if result.ok else "error")
    return result


@dataclass
class _SpawnPlan:
    """How to start the child, and what to tell it once it is running."""

    argv: list[str]
    #: Where the snapshot will be *from the child's point of view*, which is not
    #: where it is from ours once a bind mount has moved it.
    snapshot_path: str | None
    #: What this process waits, which is the analysis timeout plus whatever the
    #: backend needs to get to the first line of it.
    wall_clock: int
    #: The environment of the command being spawned — the interpreter under one
    #: backend and the container CLI under the other, which need different
    #: things. Not the environment the analysis code sees; under the container
    #: backend that is built separately and from nothing.
    env: dict[str, str]
    #: The container's name under the container backend, so the timeout path
    #: has something to kill. None under the subprocess backend, where killing
    #: the child is the same act as killing what we waited on.
    container_name: str | None = None


def _spawn_plan(snapshot_path: Path | str | None, timeout: int) -> _SpawnPlan:
    """Choose how to start the child."""
    if not container.enabled():
        return _SpawnPlan(
            # -I is isolated mode: no cwd on sys.path, no user site-packages,
            # no PYTHONPATH. The runner is invoked by path rather than as
            # `-m analyst.runner` precisely because of that last one, and it
            # imports nothing from the app so it needs neither.
            argv=[sys.executable, "-I", str(RUNNER_PATH)],
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            wall_clock=timeout,
            env=_child_env(),
        )

    name = container.container_name()
    return _SpawnPlan(
        argv=container.build_argv(snapshot_path, name),
        snapshot_path=container.CONTAINER_SNAPSHOT_PATH if snapshot_path else None,
        # Pulling and starting an image is not analysis time. Charging it to
        # the analysis would silently shorten every question by a second or so
        # and make the timeout mean something different per backend.
        wall_clock=timeout + settings.sandbox_startup_grace_seconds,
        # The full environment, because this argv is the container CLI and not
        # the analysis: it needs DOCKER_HOST, HOME for its config, and whatever
        # else the operator pointed it at. What the *analysis* may see is set
        # with --env inside build_argv, and starts empty.
        env=dict(os.environ),
        container_name=name,
    )


def _run_code(code: str, snapshot_path: Path | str | None, timeout_seconds: int | None = None) -> ExecutionResult:
    if not code or not code.strip():
        return ExecutionResult(ok=False, code=code or "", error="No code to execute.")
    if len(code) > MAX_CODE_CHARS:
        return ExecutionResult(
            ok=False,
            code=code[:MAX_CODE_CHARS],
            error=f"Code exceeds the {MAX_CODE_CHARS:,}-character limit for one execution.",
        )

    timeout = int(timeout_seconds or settings.sandbox_timeout_seconds)
    try:
        plan = _spawn_plan(snapshot_path, timeout)
    except container.SandboxUnavailable as exc:
        # Deliberately not a fallback to the subprocess backend. See the module
        # docstring in analyst/container.py: an operator who asked for a
        # boundary and quietly did not get one is the worse outcome.
        logger.error("Container sandbox unavailable: %s", exc)
        return ExecutionResult(ok=False, code=code, error=f"The analysis sandbox is not available: {exc}")

    request = {
        "code": code,
        "snapshot_path": plan.snapshot_path,
        "memory_mb": settings.sandbox_memory_mb,
        # CPU seconds slightly under the wall clock, so a busy loop trips the
        # cheaper limit first where rlimits exist. Measured against the
        # analysis's own timeout, never the container start-up grace — the
        # grace is for the runtime, not extra CPU for the code.
        "cpu_seconds": max(1, timeout - 1),
    }

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            plan.argv,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=plan.wall_clock,
            cwd=str(BACKEND_DIR),
            env=plan.env,
        )
    except subprocess.TimeoutExpired:
        duration = int((time.perf_counter() - started) * 1000)
        if plan.container_name is not None:
            # The timeout killed the client, not the container. Without this the
            # analysis that ran too long keeps running, unattended, forever.
            container.kill(plan.container_name)
        logger.info("Sandbox execution timed out after %ss", timeout)
        return ExecutionResult(
            ok=False,
            code=code,
            error=f"Execution timed out after {timeout} seconds and was stopped.",
            duration_ms=duration,
        )
    except Exception as exc:
        logger.exception("Could not start the sandbox process")
        return ExecutionResult(ok=False, code=code, error=f"Could not start the sandbox: {exc}")

    duration = int((time.perf_counter() - started) * 1000)
    payload = _parse_response(completed.stdout)
    if payload is None:
        detail = (completed.stderr or "").strip()[-800:] or "no output"
        logger.warning("Sandbox produced no parsable result: %s", detail)
        return ExecutionResult(
            ok=False,
            code=code,
            error=f"The sandbox exited without returning a result ({detail}).",
            duration_ms=duration,
        )

    return ExecutionResult(
        ok=bool(payload.get("ok")),
        code=code,
        stdout=payload.get("stdout", ""),
        stdout_truncated=bool(payload.get("stdout_truncated")),
        error=payload.get("error"),
        result=payload.get("result"),
        figures=payload.get("figures") or [],
        duration_ms=duration,
        limits=payload.get("limits", ""),
    )


def _parse_response(stdout: str) -> dict | None:
    """Take the envelope after the marker, ignoring anything printed before it."""
    if RESULT_MARKER not in stdout:
        return None
    _, _, tail = stdout.rpartition(RESULT_MARKER)
    try:
        return json.loads(tail.strip())
    except json.JSONDecodeError:
        return None
