"""Phase 8: the sandbox as a boundary rather than a convention.

`analyst/runner.py` has always been honest that its import denylist and audit
hook stop *misbehaving* generated code and not *attacking* generated code, and
every phase before this one deferred the fix. `SANDBOX_BACKEND=container` is the
fix: the same runner, in a container with no network namespace and a read-only
root.

These tests are about the configuration of that boundary, not the boundary
itself. What they can assert without a daemon is that the flags which create it
are present, that nothing silently proceeds without them, and that the analysis
code's environment is built rather than inherited. What they cannot assert is
that the kernel then honours those flags — for that there is
`ops/verify_sandbox.py`, which needs a running daemon and makes nine assertions
against a real container.

The split is deliberate and worth naming: a test suite that mocked the daemon
would prove only that this repository can spell `--network`. The claim "the
sandbox cannot reach the network" is an observation about a machine, and it is
made on a machine.
"""

import subprocess

import pytest

from analyst import container, sandbox


@pytest.fixture
def containerised(monkeypatch):
    """Switch the sandbox to the container backend for one test."""
    from config import settings

    monkeypatch.setattr(settings, "sandbox_backend", "container")
    monkeypatch.setattr(settings, "sandbox_runtime", "docker")
    monkeypatch.setattr(settings, "sandbox_image", "analyst-sandbox:test")
    # `docker` need not be installed to build a command line; only to run one.
    monkeypatch.setattr(container, "runtime_path", lambda: "/usr/bin/docker")
    return settings


# --- the flags that make it a boundary ---------------------------------------


@pytest.mark.parametrize(
    ("flag", "why"),
    [
        ("--network", "without this the sandbox shares the API's network namespace"),
        ("--read-only", "without this generated code can write its own filesystem"),
        ("--cap-drop", "without this the sandbox keeps capabilities it has no use for"),
        ("no-new-privileges", "without this a setuid binary in the image is a path to root"),
    ],
)
def test_the_isolation_flags_are_all_present(containerised, flag, why):
    argv = container.build_argv(None, "test-name")
    assert flag in argv, why


def test_the_network_is_none_and_not_merely_restricted(containerised):
    """`--network none` is the assertion; anything else is a smaller claim."""
    argv = container.build_argv(None, "test-name")
    assert argv[argv.index("--network") + 1] == "none"


def test_memory_cannot_be_evaded_through_swap(containerised):
    """--memory alone lets a process use twice it; the pair is the actual cap."""
    argv = container.build_argv(None, "test-name")
    memory = next(arg for arg in argv if arg.startswith("--memory="))
    swap = next(arg for arg in argv if arg.startswith("--memory-swap="))
    assert memory.split("=")[1] == swap.split("=")[1]


def test_a_fork_bomb_is_bounded(containerised):
    """Neither the audit hook nor a Windows rlimit stops one; --pids-limit does."""
    argv = container.build_argv(None, "test-name")
    assert any(arg.startswith("--pids-limit=") for arg in argv)


# --- what the sandbox is allowed to see --------------------------------------


def test_only_the_one_dataset_is_mounted(containerised, tmp_path):
    """Not the directory it sits in — that holds every other account's data.

    The subprocess backend runs with the whole models/ tree readable, so this
    is a gap the container backend closes rather than one it inherits.
    """
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("a\n1\n", encoding="utf-8")

    argv = container.build_argv(snapshot, "test-name")
    mounts = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--volume"]

    assert len(mounts) == 1
    host, _, mode = mounts[0].rpartition(":")
    assert mode == "ro", "the dataset is mounted writable"
    assert host.endswith("snapshot.csv"), f"a directory was mounted, not a file: {host}"


def test_the_analysis_environment_is_built_not_inherited(containerised, monkeypatch):
    """The inverse of the subprocess backend's passthrough list, and stricter.

    A passthrough list has to be updated whenever a secret is added to the
    deployment. Building the environment from nothing means a variable added
    tomorrow is absent by default rather than present by oversight.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-secret-value")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@db/analyst")

    argv = container.build_argv(None, "test-name")
    passed = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--env"]

    assert not any("secret-value" in value for value in passed)
    assert not any("password" in value for value in passed)
    assert {value.split("=")[0] for value in passed} == {
        "MPLBACKEND",
        "MPLCONFIGDIR",
        "HOME",
        "PYTHONDONTWRITEBYTECODE",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
    }


def test_the_runner_is_told_where_the_mount_landed(containerised, tmp_path):
    """The child reads the snapshot at the container's path, not at ours."""
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("a\n1\n", encoding="utf-8")

    plan = sandbox._spawn_plan(snapshot, timeout=20)
    assert plan.snapshot_path == container.CONTAINER_SNAPSHOT_PATH


# --- refusing rather than degrading ------------------------------------------


def test_a_missing_runtime_refuses_instead_of_falling_back(monkeypatch):
    """Everything else in this app degrades; a boundary must not.

    Redis falls back to inline execution and S3 to local disk, because a slow
    answer beats no answer. Here the equivalent fallback would run untrusted
    code with none of the isolation that was asked for, and say nothing.
    """
    from config import settings

    monkeypatch.setattr(settings, "sandbox_backend", "container")
    monkeypatch.setattr(container, "runtime_path", lambda: None)

    result = sandbox.run_code("1 + 1", None)

    assert not result.ok
    assert "not available" in result.error
    assert result.result is None, "the code ran anyway"


def test_an_unreadable_seccomp_profile_refuses_the_execution(containerised, monkeypatch, tmp_path):
    """Asking for a profile and running without it is the silent downgrade."""
    monkeypatch.setattr(containerised, "sandbox_seccomp_profile", str(tmp_path / "does-not-exist.json"))

    with pytest.raises(container.SandboxUnavailable, match="not a readable file"):
        container.build_argv(None, "test-name")


def test_a_seccomp_profile_that_exists_is_applied(containerised, monkeypatch, tmp_path):
    profile = tmp_path / "seccomp.json"
    profile.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(containerised, "sandbox_seccomp_profile", str(profile))

    argv = container.build_argv(None, "test-name")
    assert any(arg.startswith("seccomp=") for arg in argv)


def test_the_shipped_seccomp_profile_is_an_allowlist():
    """A denylist profile would be weaker than the default it replaces.

    A custom profile does not add to the runtime's default, it replaces it. So
    a profile written as "deny these dangerous syscalls" silently permits
    everything the default blocked and nobody thought to list — which is how a
    hardening step becomes a regression.
    """
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    profile = json.loads((repo_root / "ops" / "seccomp-analyst.json").read_text(encoding="utf-8"))

    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    allowed = {name for entry in profile["syscalls"] for name in entry["names"]}
    assert profile["syscalls"][0]["action"] == "SCMP_ACT_ALLOW"

    for escape in ("ptrace", "mount", "unshare", "setns", "bpf", "init_module", "keyctl", "socket", "userfaultfd"):
        assert escape not in allowed, f"{escape} is reachable from the analysis sandbox"

    # And it has to remain usable: CPython cannot start without these.
    for essential in ("read", "write", "mmap", "openat", "futex", "clone", "execve"):
        assert essential in allowed, f"{essential} is denied; the interpreter will not start"


# --- the default is unchanged -------------------------------------------------


def test_the_subprocess_backend_is_still_the_default():
    """Phase 6's rule: a capability that changes behaviour is opt-in.

    The container backend needs an image built and a runtime reachable. Making
    it the default would break every install that has neither, to protect
    against a user who does not exist on a single-operator machine.
    """
    from config import Settings

    assert Settings().sandbox_backend == "subprocess"
    assert not container.enabled()


def test_the_subprocess_backend_still_runs_code(tmp_path):
    """The boundary work must not have broken the path everyone is using."""
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_text("a,b\n1,4\n2,5\n3,6\n", encoding="utf-8")

    result = sandbox.run_code("int(df['b'].sum())", snapshot)

    assert result.ok, result.error
    assert result.result["value"] == 15


def test_readyz_says_which_sandbox_is_in_force(client):
    """An operator should not have to read config.py to answer this."""
    body = client.get("/readyz").json()
    sandbox_check = body["checks"]["sandbox"]

    assert sandbox_check["backend"] == "subprocess"
    assert sandbox_check["network_isolated"] is False
    assert "SANDBOX_BACKEND" in sandbox_check["note"]


def test_readyz_flags_a_boundary_that_was_asked_for_and_is_absent(client, monkeypatch):
    """Configured-but-missing is the case worth finding from a health check."""
    from config import settings

    monkeypatch.setattr(settings, "sandbox_backend", "container")
    monkeypatch.setattr(container, "runtime_path", lambda: None)

    sandbox_check = client.get("/readyz").json()["checks"]["sandbox"]

    assert sandbox_check["ok"] is False
    assert "not on PATH" in sandbox_check["note"]


def test_readyz_flags_a_seccomp_profile_that_cannot_be_read(client, monkeypatch, tmp_path):
    """The other way a configured boundary is absent, and it fails identically.

    `build_argv` raises on an unreadable profile exactly as it does on a missing
    runtime, so both refuse every execution. A readiness check that reported
    only the first would call a deployment ready while it answered nothing.
    """
    from config import settings

    monkeypatch.setattr(settings, "sandbox_backend", "container")
    monkeypatch.setattr(settings, "sandbox_seccomp_profile", str(tmp_path / "gone.json"))
    monkeypatch.setattr(container, "runtime_path", lambda: "/usr/bin/docker")

    sandbox_check = client.get("/readyz").json()["checks"]["sandbox"]

    assert sandbox_check["ok"] is False
    assert sandbox_check["seccomp_profile_readable"] is False
    assert "cannot be read" in sandbox_check["note"]


# --- the timeout path ---------------------------------------------------------


def test_a_timed_out_container_is_killed_not_merely_abandoned(containerised, monkeypatch):
    """`subprocess.run(timeout=...)` kills the client; the container runs on.

    With `--rm` and nobody watching, that container keeps burning a CPU until
    the daemon restarts — the failure that reads as the machine getting slower
    over a week for no reason.
    """
    killed: list[str] = []

    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1)

    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    monkeypatch.setattr(container, "kill", killed.append)

    result = sandbox.run_code("while True:\n    pass\n", None)

    assert not result.ok
    assert len(killed) == 1
    assert killed[0].startswith("analyst-sandbox-")
