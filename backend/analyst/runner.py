"""The child process that actually executes analysis code.

Run as `python -m analyst.runner`, reading one JSON request on stdin and
writing one JSON response on stdout. It never imports the app: it is a plain
Python process holding a DataFrame, so nothing it does can touch the API,
the database, or the model artifacts.

## What this is, and what it is not

This is a *restricted execution environment*, not a security boundary against
a determined attacker. It defends against the realistic threat — generated
code that loops forever, allocates everything, writes files, or phones home —
with four layers:

1. **Process isolation.** A separate interpreter, killed by the parent on
   timeout. Nothing it corrupts survives into the server.
2. **Resource limits.** Address space and CPU seconds via `setrlimit` on
   POSIX. Windows has no equivalent, so there the wall-clock timeout and the
   output caps are the only ceilings — stated plainly rather than implied.
3. **An import denylist**, so the obvious escapes are not even importable.
4. **An audit hook** (PEP 578) refusing network, subprocess, and file writes.
   Audit hooks cannot be uninstalled once added, so code cannot lift them.

A real adversary with arbitrary Python can eventually defeat 3 and 4. If this
is ever exposed to untrusted users, the process belongs in a container with a
seccomp profile and no network namespace — the layers here are the floor, not
the ceiling.
"""

import ast
import base64
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

# The parent finds the response after this marker, so anything user code writes
# straight to fd 1 cannot be mistaken for the envelope.
RESULT_MARKER = "<<<ANALYST_RESULT>>>"

MAX_STDOUT_CHARS = 20_000
MAX_TABLE_ROWS = 200
MAX_FIGURES = 4

# Modules with no place in a data-analysis snippet, each an obvious escape.
BLOCKED_MODULES = frozenset(
    {
        "socket",
        "ssl",
        "http",
        "urllib",
        "urllib3",
        "requests",
        "httpx",
        "ftplib",
        "smtplib",
        "telnetlib",
        "xmlrpc",
        "asyncio",
        "subprocess",
        "multiprocessing",
        "ctypes",
        "cffi",
        "shutil",
        "tempfile",
        "pickle",
        "dill",
        "joblib",
        "importlib",
        "runpy",
        "pty",
        "signal",
        "webbrowser",
        "sqlite3",
        "sqlalchemy",
        "sqlmodel",
    }
)

# Audited events that are simply not part of analysing a dataframe.
#
# Unlike the import guard, an audit hook fires wherever the operation happens,
# including deep inside a library — so this list holds only operations no
# scientific library performs on its own behalf. `ctypes.dlopen` was here and
# had to come out: scipy and scikit-learn load their native extensions through
# it at import time, so blocking it broke the two libraries most worth having.
# Direct `import ctypes` from analysis code is still refused above.
BLOCKED_AUDIT_EVENTS = (
    "socket.connect",
    "socket.bind",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "os.posix_spawn",
    "shutil.move",
    "shutil.rmtree",
    "webbrowser.open",
)


class SandboxViolation(Exception):
    """Raised inside the child when code reaches for something it may not."""


def apply_resource_limits(memory_mb: int, cpu_seconds: int) -> str:
    """Cap address space and CPU time. Returns a note on what was enforced."""
    try:
        import resource
    except ImportError:  # Windows
        return "no-rlimit"

    try:
        limit_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        # A sandbox has no business writing files at all.
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        return "rlimit"
    except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
        return f"rlimit-failed: {exc}"


def install_import_guard() -> None:
    """Refuse the blocked modules when *analysis code* imports them.

    The check is scoped to the caller's frame on purpose. A blanket denylist
    breaks the sandbox instead of securing it: matplotlib imports `importlib`,
    scikit-learn imports `joblib` and `pickle`, pandas reaches for `ctypes` —
    all internally, all legitimately. Blocking those bricks plotting and
    modelling while stopping no attacker, since the audit hook already refuses
    the *operations* that matter regardless of who imported what.

    So: an import written in the analysis snippet is subject to the denylist;
    an import made by a library on its own behalf is not.
    """
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in BLOCKED_MODULES and _called_from_analysis():
            raise SandboxViolation(
                f"Importing '{root}' is not allowed in the analysis sandbox. "
                "Use the data-analysis libraries instead (pandas, numpy, scipy, sklearn, matplotlib)."
            )
        return real_import(name, globals, locals, fromlist, level)

    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = guarded_import
    else:
        __builtins__.__import__ = guarded_import


def _called_from_analysis() -> bool:
    """True when the immediate caller is the executed snippet, not a library."""
    try:
        return sys._getframe(2).f_code.co_filename == "<analysis>"
    except ValueError:  # pragma: no cover - shallower stack than expected
        return False


def install_audit_hook() -> None:
    """Deny network, subprocess, and any file opened for writing."""

    def hook(event: str, args):
        if event in BLOCKED_AUDIT_EVENTS:
            raise SandboxViolation(f"'{event}' is not allowed in the analysis sandbox.")
        if event == "open":
            # args = (path, mode, flags); mode is None for os.open.
            mode = args[1] if len(args) > 1 else None
            if mode and any(flag in str(mode) for flag in ("w", "a", "x", "+")):
                raise SandboxViolation("Writing files is not allowed in the analysis sandbox.")

    sys.addaudithook(hook)


def _frame_to_table(frame, index_label: str | None = None) -> dict:
    """A DataFrame as {columns, rows, shape, truncated} — JSON, not a repr."""
    import pandas as pd

    truncated = len(frame) > MAX_TABLE_ROWS
    view = frame.head(MAX_TABLE_ROWS)
    if index_label is not None:
        view = view.reset_index().rename(columns={"index": index_label})
    return {
        "kind": "table",
        "columns": [str(c) for c in view.columns],
        "rows": [[_jsonable(v) for v in row] for row in view.itertuples(index=False, name=None)],
        "shape": [int(frame.shape[0]), int(frame.shape[1]) if hasattr(frame, "shape") and frame.ndim > 1 else 1],
        "truncated": bool(truncated),
        "_pd": pd.__version__,
    }


def _jsonable(value):
    import numpy as np
    import pandas as pd

    if value is None or isinstance(value, (bool, int, float, str)):
        return None if isinstance(value, float) and (value != value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NaT or (hasattr(pd, "isna") and not hasattr(value, "__len__") and pd.isna(value)):
        return None
    return str(value)


def _capture_result(value) -> dict | None:
    """Turn the value of the final expression into something renderable."""
    if value is None:
        return None
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return _frame_to_table(value)
    if isinstance(value, pd.Series):
        return _frame_to_table(value.to_frame(name=value.name or "value"), index_label="index")
    if isinstance(value, (int, float, bool, str)):
        return {"kind": "scalar", "value": _jsonable(value)}
    return {"kind": "text", "value": repr(value)[:4000]}


def _collect_figures() -> list[dict]:
    """Any matplotlib figure the code drew, as base64 PNG.

    Rendered to an in-memory buffer rather than a file, which is also why the
    audit hook can refuse every write without breaking plotting.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib ships with the app
        return []

    images = []
    for number in plt.get_fignums()[:MAX_FIGURES]:
        figure = plt.figure(number)
        if not figure.get_axes():
            continue
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor="#0f172a")
        images.append({"kind": "image", "mime": "image/png", "data": base64.b64encode(buffer.getvalue()).decode()})
    plt.close("all")
    return images


def _split_last_expression(code: str):
    """Compile the body and the trailing expression separately.

    Notebook semantics: the value of a final bare expression is the result,
    without the user having to remember to print it.
    """
    tree = ast.parse(code, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body[-1].value)
        body = ast.Module(body=tree.body[:-1], type_ignores=[])
        return compile(body, "<analysis>", "exec"), compile(last, "<analysis>", "eval")
    return compile(tree, "<analysis>", "exec"), None


def build_namespace(snapshot_path: str | None):
    """The globals analysis code runs against: df plus the usual libraries."""
    import numpy as np
    import pandas as pd

    namespace: dict = {"pd": pd, "np": np}

    try:
        import matplotlib

        matplotlib.use("Agg")  # no display, no window server
        import matplotlib.pyplot as plt

        plt.style.use("dark_background")
        namespace["plt"] = plt
    except ImportError:  # pragma: no cover
        pass

    if snapshot_path and os.path.exists(snapshot_path):
        frame = pd.read_csv(snapshot_path)
        namespace["df"] = frame
        namespace["data"] = frame
    return namespace


def execute(request: dict) -> dict:
    # Order matters. The namespace is built first because reading the snapshot
    # and letting matplotlib build its font cache both need the file access the
    # guards are about to remove -- and RLIMIT_FSIZE=0 would block that cache
    # write too. Only our own setup runs before the guards; no request code does.
    try:
        namespace = build_namespace(request.get("snapshot_path"))
    except Exception as exc:
        return {"ok": False, "error": f"Could not load the dataset snapshot: {exc}", "limits": "not-applied"}

    limits = apply_resource_limits(
        memory_mb=int(request.get("memory_mb", 1024)),
        cpu_seconds=int(request.get("cpu_seconds", 30)),
    )
    install_import_guard()
    install_audit_hook()

    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    result = None
    error = None

    try:
        body, tail = _split_last_expression(request["code"])
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(body, namespace)  # noqa: S102 - executing analysis code is the point
            if tail is not None:
                result = _capture_result(eval(tail, namespace))  # noqa: S307
    except SandboxViolation as exc:
        error = str(exc)
    except SyntaxError as exc:
        error = f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except MemoryError:
        error = "The code ran out of memory inside the sandbox."
    except Exception:
        # Only the user frames: the runner's own stack is noise to them.
        frames = traceback.format_exception(*sys.exc_info())
        error = "".join(frames[-2:]).strip()

    stdout = stdout_buffer.getvalue()
    truncated_stdout = len(stdout) > MAX_STDOUT_CHARS

    return {
        "ok": error is None,
        "error": error,
        "stdout": stdout[:MAX_STDOUT_CHARS],
        "stdout_truncated": truncated_stdout,
        "stderr": stderr_buffer.getvalue()[:2000],
        "result": result,
        "figures": _collect_figures() if error is None else [],
        "limits": limits,
    }


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
    except Exception as exc:  # pragma: no cover - malformed parent request
        payload = {"ok": False, "error": f"Bad sandbox request: {exc}"}
    else:
        payload = execute(request)

    sys.stdout.write(f"\n{RESULT_MARKER}\n")
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
