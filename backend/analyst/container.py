"""Running the analyst sandbox inside a container, where the boundary is real.

`analyst/runner.py` says plainly what it is: an import denylist and a PEP 578
audit hook, which stop generated code that *misbehaves* and would not stop
generated code that *attacks*. Both live inside the interpreter they are
guarding, and anything with arbitrary Python eventually gets underneath them.
Every phase so far has been honest about that and deferred the fix. This is the
fix.

The same runner, unchanged, is executed inside a container with:

* **no network namespace at all** (`--network none`) — not a denylist of
  socket calls that has to anticipate every way of opening one, but an
  interface list with nothing in it but loopback. Exfiltration stops being a
  question about the audit hook and becomes a question about the kernel.
* **a read-only root filesystem**, with one small tmpfs for the scratch that
  matplotlib insists on. `RLIMIT_FSIZE` already made writes fail; this makes
  them fail even if the limit is lifted.
* **every capability dropped** and `no-new-privileges`, so a setuid binary
  inside the image is not a path anywhere.
* **a seccomp profile**, so the syscalls a data-analysis process never makes
  are not reachable to be exploited.
* **exactly one dataset mounted**, read-only, at a fixed path. The subprocess
  backend runs with the whole `models/` tree readable; here an execution can
  see the snapshot it was given and no other account's data, which closes a gap
  the interpreter guards never addressed at all.

What this costs, stated rather than buried:

* **A container start per execution**, about a second. `sandbox_startup_grace_
  seconds` is added to the caller's timeout so the analysis still gets the time
  it was promised.
* **Access to a container runtime.** When the API is itself a container, that
  means a mounted docker socket — which is root on the host, and therefore
  moves risk rather than removing it. Podman's rootless socket is the better
  answer and takes the same flags. `docs/security.md` works through both.

There is deliberately no fallback to the subprocess backend when the runtime is
missing. Elsewhere in this application an unreachable dependency degrades:
Redis falls back to inline, S3 to local disk. Those are availability features
and a slower answer beats no answer. This is a boundary, and silently answering
without it would mean an operator who asked for isolation not getting it and
not being told — the one failure mode worth an outage.
"""

import logging
import secrets
import shutil
import subprocess
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

#: Where the one dataset an execution may see is mounted. The runner is told
#: this path in its request, so it needs no knowledge of being containerised.
CONTAINER_SNAPSHOT_PATH = "/data/snapshot.csv"

#: Where ops/Dockerfile.sandbox puts the runner.
CONTAINER_RUNNER_PATH = "/app/runner.py"

#: Scratch for matplotlib's font cache, which it builds before it will draw.
CONTAINER_TMPFS = "/tmp"


class SandboxUnavailable(RuntimeError):
    """The container backend was asked for and cannot be provided."""


def enabled() -> bool:
    return settings.sandbox_backend.strip().lower() == "container"


def runtime_path() -> str | None:
    """The container runtime executable, or None when there is not one."""
    return shutil.which(settings.sandbox_runtime)


def host_snapshot_path(snapshot_path: Path | str) -> str:
    """The path to hand the container daemon for a bind mount.

    Bind mounts are resolved by the daemon, against the filesystem the *daemon*
    can see. When the API runs on the host those are the same filesystem and
    this is the identity function. When the API is a container talking to the
    host's daemon they are not, and mounting the path the API sees produces an
    empty directory rather than an error — the failure that reads as "the
    dataset vanished" and has nothing to do with datasets.

    `SANDBOX_HOST_MODELS_DIR` names the host directory behind the API's
    `models/` mount, and the prefix is swapped for it.
    """
    from utils.helpers import BASE_DIR

    resolved = Path(snapshot_path).resolve()
    host_models = (settings.sandbox_host_models_dir or "").strip()
    if not host_models:
        return str(resolved)

    models_dir = (BASE_DIR / "models").resolve()
    try:
        relative = resolved.relative_to(models_dir)
    except ValueError:
        # Outside models/ — a test fixture or a bespoke path. Nothing to
        # rewrite, and guessing would be worse than passing it through.
        return str(resolved)
    # Posix separators: the daemon this is being sent to is the one that
    # resolves it, and it is Linux even when this process is not.
    return f"{host_models.rstrip('/')}/{relative.as_posix()}"


def container_name() -> str:
    """A unique name for one execution's container, so it can be killed by name.

    Needed because `subprocess.run(timeout=...)` kills the *client*, and the
    container it asked the daemon to start carries on running with `--rm` set
    and nobody watching it. Without a name there is nothing to aim `kill` at,
    and a timed-out analysis would leak a container burning a CPU until the
    daemon is restarted — the failure that looks like the machine getting
    slower over a week for no reason.
    """
    return f"analyst-sandbox-{secrets.token_hex(8)}"


def kill(name: str) -> None:
    """Stop a container by name, best effort.

    Never raises. This runs on the timeout path, where an exception would
    replace "your analysis took too long" with a stack trace about Docker.
    """
    runtime = runtime_path()
    if runtime is None:  # pragma: no cover - it started, so it was there
        return
    try:
        subprocess.run(
            [runtime, "kill", name],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:  # pragma: no cover - the daemon went away mid-execution
        logger.warning("Could not kill sandbox container %s", name, exc_info=True)


def build_argv(snapshot_path: Path | str | None, name: str) -> list[str]:
    """The full command line for one execution.

    Every flag here removes something. If one has to be taken out to make a
    workload run, that is a decision to write down in `docs/security.md`, not a
    default to quietly relax.
    """
    runtime = runtime_path()
    if runtime is None:
        raise SandboxUnavailable(
            f"SANDBOX_BACKEND=container needs '{settings.sandbox_runtime}' on PATH, and it is not there."
        )

    argv = [
        runtime,
        "run",
        "--rm",
        # stdin carries the request; there is no tty and nothing interactive.
        "--interactive",
        "--name",
        name,
        # The whole point. Loopback only, no route off the machine, no DNS.
        "--network",
        "none",
        "--read-only",
        # Small, and noexec so a payload written here cannot then be run.
        f"--tmpfs={CONTAINER_TMPFS}:rw,noexec,nosuid,size=64m,mode=1777",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        # Memory is capped equal to swap so the limit cannot be evaded by
        # swapping: --memory alone lets a process use twice it in total.
        f"--memory={settings.sandbox_memory_mb}m",
        f"--memory-swap={settings.sandbox_memory_mb}m",
        f"--pids-limit={settings.sandbox_pids_limit}",
        f"--cpus={settings.sandbox_cpus}",
    ]

    profile = (settings.sandbox_seccomp_profile or "").strip()
    if profile:
        path = Path(profile)
        if not path.is_file():
            raise SandboxUnavailable(
                f"SANDBOX_SECCOMP_PROFILE points at {profile}, which is not a readable file. "
                "Refusing to run with less isolation than was asked for."
            )
        argv += ["--security-opt", f"seccomp={path.resolve()}"]

    for name, value in _child_env().items():
        argv += ["--env", f"{name}={value}"]

    if snapshot_path is not None:
        # One file, read-only. Not the directory it sits in: that directory
        # holds every other account's snapshots too.
        argv += ["--volume", f"{host_snapshot_path(snapshot_path)}:{CONTAINER_SNAPSHOT_PATH}:ro"]

    # The entrypoint is given explicitly rather than relied upon from the
    # image, so pointing SANDBOX_IMAGE at a plain python image with the
    # dependencies installed also works. -I is isolated mode, exactly as the
    # subprocess backend uses it.
    argv += ["--entrypoint", "python", settings.sandbox_image, "-I", CONTAINER_RUNNER_PATH]
    return argv


def _child_env() -> dict[str, str]:
    """The environment inside the container.

    Built from nothing rather than filtered from this process's, which is the
    inverse of the subprocess backend's passthrough list and strictly safer: a
    variable added to the API's environment later cannot leak in here by
    default. The image supplies PATH itself.
    """
    return {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": f"{CONTAINER_TMPFS}/mpl",
        "HOME": CONTAINER_TMPFS,
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _container_note(runtime: str | None, profile: str | None, profile_readable: bool) -> str:
    """One sentence an operator can act on, for whichever thing is wrong."""
    if runtime is None:
        return (
            f"SANDBOX_BACKEND=container, but '{settings.sandbox_runtime}' is not on PATH. "
            "Every execution will fail rather than silently run unisolated."
        )
    if not profile_readable:
        return (
            f"SANDBOX_SECCOMP_PROFILE points at {profile}, which cannot be read. "
            "Every execution will fail rather than run with less isolation than was asked for."
        )
    return "Generated code runs in a container with no network namespace and a read-only root."


def describe() -> dict:
    """What isolation is actually in force — for /readyz and the config page.

    Reported rather than asserted. An operator who set SANDBOX_BACKEND and never
    built the image should find that out from a health check, not from the first
    question a user asks.
    """
    if not enabled():
        return {
            # Not asked for, so nothing is missing. `ok` here means "the
            # isolation this deployment asked for is present", which is
            # trivially true of a deployment that asked for none — reporting
            # false would make every default install look broken.
            "ok": True,
            "backend": "subprocess",
            "network_isolated": False,
            "read_only_root": False,
            "seccomp_profile": None,
            "note": (
                "Generated code runs in a restricted interpreter in this process's container, not "
                "behind a kernel boundary. Adequate for trusted operators; set SANDBOX_BACKEND="
                "container before untrusted users can reach the analyst."
            ),
        }

    runtime = runtime_path()
    profile = (settings.sandbox_seccomp_profile or "").strip() or None
    profile_readable = profile is None or Path(profile).is_file()
    return {
        # Both failures refuse every execution, so both belong in one verdict.
        # An unreadable seccomp profile is as fatal as a missing runtime —
        # build_argv raises on either — and reporting only the first would make
        # a deployment that fails every analysis look ready.
        "ok": runtime is not None and profile_readable,
        "backend": "container",
        "runtime": settings.sandbox_runtime,
        "runtime_found": runtime is not None,
        "image": settings.sandbox_image,
        "network_isolated": True,
        "read_only_root": True,
        "seccomp_profile": profile,
        "seccomp_profile_readable": profile_readable,
        "note": _container_note(runtime, profile, profile_readable),
    }
