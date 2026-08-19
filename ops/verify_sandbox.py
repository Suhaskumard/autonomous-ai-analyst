"""Prove the containerised sandbox is actually a boundary — Phase 8.

    python ops/verify_sandbox.py [--build] [--runtime docker] [--seccomp ops/seccomp-analyst.json]

`SANDBOX_BACKEND=container` is a configuration claim. This is the check that
turns it into an observation, and it exists because every part of it can be
true in the config and false on the machine: the image may not be built, the
seccomp profile may deny a syscall numpy needs, the bind mount may resolve to
nothing. Each of those fails in a different and confusing way at the first user
question, and in an obvious way here.

Nine assertions, in the order in which failing one makes the next meaningless:

    1  the runtime is present and its daemon answers
    2  the sandbox image exists
    3  a trivial expression evaluates                (the runner works at all)
    4  pandas, numpy, scipy and sklearn import       (the seccomp profile is not
                                                      too narrow to be usable)
    5  a mounted snapshot is readable and correct    (the bind mount resolves)
    6  the network is unreachable                    (--network none)
    7  the root filesystem refuses writes            (--read-only)
    8  /tmp accepts a write                          (matplotlib can still draw)
    9  a runaway loop is stopped, and its container  (the timeout kills what it
       does not survive it                            started, not just the client)

Assertions 6 and 7 are the ones that matter, and they are written to FAIL if
the isolation is absent — a sandbox that can reach the network passes nothing
here. That direction is deliberate: a check that only proves the happy path
would pass just as well against the subprocess backend, which is exactly the
mistake this is meant to catch.

Run it after building the image, after changing the seccomp profile, and after
any upgrade of the container runtime. It leaves nothing behind.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
DOCKERFILE = REPO_ROOT / "ops" / "Dockerfile.sandbox"

# Long enough for a cold image with numpy and sklearn to import.
TIMEOUT = 120


class Failure(Exception):
    """Something the container backend promises is not true on this machine."""


def _configure(runtime: str, image: str, seccomp: str | None) -> None:
    """Point the app's own settings at what we are testing.

    Imported and configured rather than shelled out to, so this exercises the
    same `analyst.sandbox.run_code` the API calls. A drill that builds its own
    docker command line proves the drill works.
    """
    sys.path.insert(0, str(BACKEND_DIR))
    from config import settings

    settings.sandbox_backend = "container"
    settings.sandbox_runtime = runtime
    settings.sandbox_image = image
    settings.sandbox_seccomp_profile = seccomp
    # Short, because assertion 9 waits out a deliberate infinite loop.
    settings.sandbox_timeout_seconds = 15


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
    if not condition:
        raise Failure(f"{name}: {detail or 'assertion failed'}")


def runtime_available(runtime: str) -> None:
    print("1. runtime")
    try:
        done = subprocess.run([runtime, "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True)
    except FileNotFoundError:
        raise Failure(f"{runtime!r} is not on PATH.") from None
    check(f"{runtime} daemon answers", done.returncode == 0, (done.stderr or "").strip()[:200])
    print(f"        server {done.stdout.strip()}")


def image_present(runtime: str, image: str, build: bool) -> None:
    print("2. image")
    if build:
        print(f"        building {image} from {DOCKERFILE.relative_to(REPO_ROOT)} ...")
        done = subprocess.run(
            [runtime, "build", "-f", str(DOCKERFILE), "-t", image, str(BACKEND_DIR)],
            capture_output=True,
            text=True,
        )
        check("image builds", done.returncode == 0, (done.stderr or "").strip()[-400:])

    done = subprocess.run([runtime, "image", "inspect", image], capture_output=True, text=True)
    check(
        f"{image} exists",
        done.returncode == 0,
        "not built — re-run with --build" if done.returncode else "",
    )


def runner_works(run_code) -> None:
    print("3. the runner responds")
    result = run_code("21 * 2", None)
    check("a trivial expression evaluates", result.ok, result.error or "")
    check("and returns the right answer", (result.result or {}).get("value") == 42, str(result.result))


def libraries_import(run_code) -> None:
    print("4. the analysis libraries are usable")
    code = (
        "import numpy, pandas, scipy, sklearn\n"
        "import scipy.stats, sklearn.linear_model\n"
        "float(scipy.stats.norm.cdf(0)) + float(numpy.array([1.0, 2.0]).mean())"
    )
    result = run_code(code, None)
    # The assertion the seccomp profile is most likely to break: these libraries
    # load native extensions at import, which is where a missing syscall shows.
    check("numpy/pandas/scipy/sklearn import and compute", result.ok, (result.error or "")[-400:])


def snapshot_mounts(run_code) -> None:
    print("5. the dataset mount resolves")
    with tempfile.TemporaryDirectory() as tmp:
        snapshot = Path(tmp) / "snapshot.csv"
        snapshot.write_text("a,b\n1,4\n2,5\n3,6\n", encoding="utf-8")
        result = run_code("int(df['b'].sum())", snapshot)

    check("the mounted snapshot is readable", result.ok, (result.error or "")[-300:])
    check(
        "and holds the rows that were written",
        (result.result or {}).get("value") == 15,
        f"got {result.result} — an empty mount usually means SANDBOX_HOST_MODELS_DIR is wrong",
    )


def _probe(code: str) -> subprocess.CompletedProcess:
    """Run a bare snippet in the sandbox container, bypassing the runner.

    Deliberately not through `run_code`. The runner would refuse both of the
    probes below on its own — `import socket` is on its denylist and `open(...,
    "w")` trips its audit hook — so a probe that went through it would pass just
    as happily under the subprocess backend and prove nothing about the
    container. These two assertions are about the kernel, so they have to reach
    it: same flags, same image, different command.
    """
    from analyst import container

    argv = container.build_argv(None, container.container_name())
    # build_argv ends with `--entrypoint python <image> -I /app/runner.py`.
    # Keep everything through the image and substitute the command.
    assert argv[-2:] == ["-I", container.CONTAINER_RUNNER_PATH]
    return subprocess.run(argv[:-2] + ["-c", code], capture_output=True, text=True, timeout=TIMEOUT)


def network_is_gone() -> None:
    print("6. no network namespace")
    code = "; ".join(
        [
            "import socket",
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
            "s.settimeout(5)",
            "s.connect(('1.1.1.1', 53))",
            "print('CONNECTED')",
        ]
    )
    done = _probe(code)
    check(
        "an outbound connection fails",
        "CONNECTED" not in done.stdout,
        "the sandbox reached the internet — --network none is not in effect",
    )
    detail = (done.stderr or "").strip().splitlines()
    print(f"        refused with: {detail[-1][:120] if detail else 'no route'}")


def root_is_read_only() -> None:
    print("7. read-only root filesystem")
    done = _probe("open('/app/written-by-the-sandbox', 'w').write('x'); print('WROTE')")
    check(
        "a write to / is refused",
        "WROTE" not in done.stdout,
        "the sandbox wrote to its own root filesystem — --read-only is not in effect",
    )
    # And the same write under /tmp must succeed, or assertion 8 has no chance.
    done = _probe("open('/tmp/scratch', 'w').write('x'); print('WROTE')")
    check("a write to /tmp is allowed", "WROTE" in done.stdout, (done.stderr or "").strip()[-200:])


def tmp_is_writable(run_code) -> None:
    print("8. the scratch tmpfs still works")
    # Not a security assertion — a usability one. matplotlib builds a font cache
    # before it will render, and a sandbox where that fails draws no figures.
    result = run_code("import matplotlib.pyplot as plt; plt.plot([1,2,3]); 'DREW'", None)
    check(
        "matplotlib renders",
        result.ok and (result.result or {}).get("value") == "DREW",
        (result.error or "")[-300:] or "no figure",
    )


def runaway_is_stopped(run_code, runtime: str) -> None:
    print("9. a runaway execution is stopped")
    before = _running_sandboxes(runtime)
    result = run_code("while True:\n    pass\n", None)
    check("an infinite loop is timed out", not result.ok, "it returned successfully")
    check("and reported as a timeout", "timed out" in (result.error or "").lower(), result.error or "")

    after = _running_sandboxes(runtime)
    leaked = after - before
    check(
        "no container outlives the timeout",
        not leaked,
        f"still running: {', '.join(sorted(leaked))} — the client was killed but the container was not",
    )


def _running_sandboxes(runtime: str) -> set[str]:
    done = subprocess.run(
        [runtime, "ps", "--filter", "name=analyst-sandbox-", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {line.strip() for line in done.stdout.splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runtime", default="docker", help="docker, or podman for the rootless path")
    parser.add_argument("--image", default="analyst-sandbox:latest")
    parser.add_argument("--build", action="store_true", help="build the image first")
    parser.add_argument(
        "--seccomp",
        default=None,
        help="path to a seccomp profile; omit to use the runtime's default (ops/seccomp-analyst.json is the tight one)",
    )
    args = parser.parse_args()

    if args.seccomp and not Path(args.seccomp).is_file():
        print(f"No seccomp profile at {args.seccomp}", file=sys.stderr)
        return 2

    print(f"Verifying the container sandbox: runtime={args.runtime} image={args.image} seccomp={args.seccomp}\n")

    try:
        runtime_available(args.runtime)
        image_present(args.runtime, args.image, args.build)

        _configure(args.runtime, args.image, args.seccomp)
        from analyst.sandbox import run_code

        runner_works(run_code)
        libraries_import(run_code)
        snapshot_mounts(run_code)
        network_is_gone()
        root_is_read_only()
        tmp_is_writable(run_code)
        runaway_is_stopped(run_code, args.runtime)
    except Failure as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        print(
            "\nThe container backend is not safe to enable until this passes. Leave SANDBOX_BACKEND unset "
            "rather than running with isolation that is configured and not present.",
            file=sys.stderr,
        )
        return 1

    print("\nAll nine assertions hold. SANDBOX_BACKEND=container is doing what it claims on this machine.")
    print(json.dumps({"runtime": args.runtime, "image": args.image, "seccomp": args.seccomp}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
