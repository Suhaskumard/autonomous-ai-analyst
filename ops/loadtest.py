"""Load test the upload endpoint and one training run, and record the numbers.

    python ops/loadtest.py --concurrency 8 --requests 24 --json ops/results/loadtest.json

Queue depth and training duration have been exported since Phase 6, which makes
them observable but not yet meaningful: nothing had ever measured what they read
under load, so there was no number to compare an incident against. This produces
that number.

Two things are measured, because they fail differently:

* **Upload accept latency** — how long the API takes to stream a file to disk,
  hash it, and hand the job over. This is request-path work and it competes with
  every other request; it is the number that degrades first.
* **Training wall time** — how long one run takes end to end. On the inline
  backend this is also request-path work, which is the point of the comparison:
  run it inline and again with the queue, and the difference is what the queue
  bought.

Each virtual user uploads a *different* dataset, and each invocation generates a
different set again. Uploads are content-addressed: sending a file the server has
already trained on returns the stored artifacts in about a second, so a repeat
run with fixed seeds would measure the cache rather than the pipeline — a mistake
that makes any load test look wonderful. Pass `--seed` to reproduce one on purpose.

Standard library only. It reports errors as errors rather than averaging them
away; a p95 computed over requests that mostly failed says nothing.
"""

import argparse
import io
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def synthetic_csv(rows: int, seed: int) -> bytes:
    """A dataset with a learnable signal, unique per seed.

    Unique so the content hash differs and every upload is a real run; learnable
    so training does the work it would do on a real dataset instead of bailing
    out early on a target it cannot fit.
    """
    rng = random.Random(seed)
    buffer = io.StringIO()
    buffer.write("age,spend,tenure,region,churn\n")
    regions = ("north", "south", "east", "west")
    for _ in range(rows):
        age = rng.randint(18, 80)
        spend = round(rng.uniform(0, 100) + seed % 7, 2)
        tenure = rng.randint(0, 120)
        region = rng.choice(regions)
        churn = "yes" if (spend > 55 and tenure < 40) or rng.random() < 0.05 else "no"
        buffer.write(f"{age},{spend},{tenure},{region},{churn}\n")
    return buffer.getvalue().encode("utf-8")


def upload(base: str, payload: bytes, name: str, api_key: str | None, timeout: float) -> tuple[float, str | None, str]:
    """POST one dataset. Returns (seconds, job_id or None, outcome)."""
    boundary = f"----analyst{uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode(),
            b"Content-Type: text/csv\r\n\r\n",
            payload,
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="mode"\r\n\r\nauto\r\n',
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{base}/api/upload", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            answer = json.loads(reply.read().decode("utf-8"))
        return time.perf_counter() - started, answer.get("job_id"), "accepted"
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - started, None, f"http {exc.code}"
    except Exception as exc:  # timeouts, resets, refused connections
        return time.perf_counter() - started, None, type(exc).__name__


def poll_until_done(base: str, job_id: str, api_key: str | None, timeout: float) -> tuple[float, str]:
    """Wait for one job. Returns (seconds, final state)."""
    started = time.perf_counter()
    deadline = started + timeout
    while time.perf_counter() < deadline:
        request = urllib.request.Request(f"{base}/api/upload/status/{job_id}")
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=30) as reply:
                status = json.loads(reply.read().decode("utf-8"))
        except Exception:
            time.sleep(1)
            continue
        if status["state"] in ("completed", "failed"):
            return time.perf_counter() - started, status["state"]
        time.sleep(1)
    return time.perf_counter() - started, "timeout"


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile — no interpolation, so it is always an observation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered) + 0.5) - 1))
    return ordered[index]


def summarise(durations: list[float]) -> dict:
    if not durations:
        return {}
    return {
        "count": len(durations),
        "min_s": round(min(durations), 3),
        "p50_s": round(percentile(durations, 0.50), 3),
        "p95_s": round(percentile(durations, 0.95), 3),
        "max_s": round(max(durations), 3),
        "mean_s": round(statistics.fmean(durations), 3),
    }


def readyz(base: str, api_key: str | None) -> dict:
    request = urllib.request.Request(f"{base}/readyz")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=30) as reply:
        return json.loads(reply.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=4, help="uploads in flight at once")
    parser.add_argument("--requests", type=int, default=12, help="uploads in total")
    parser.add_argument("--rows", type=int, default=2000, help="rows per generated dataset")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed for the generated datasets (default: random). Reusing a seed reuses the datasets, "
        "which the server has already trained — that measures the cache, not the pipeline.",
    )
    parser.add_argument("--timeout", type=float, default=300, help="seconds before one upload is given up on")
    parser.add_argument("--train-timeout", type=float, default=1800, help="seconds to wait for the tracked run")
    parser.add_argument("--json", type=Path, default=None, help="write the full result here")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    ready = readyz(base, args.api_key)
    environment = {
        "queue": ready["checks"]["queue"],
        "storage": ready["checks"]["storage"]["backend"],
        "database": ready["checks"]["database"].get("dialect"),
        "schema": ready["checks"]["schema"]["revision"],
    }
    print(f"target      {base}")
    print(f"environment {json.dumps(environment)}")
    print(f"load        {args.requests} uploads, {args.concurrency} at a time, {args.rows} rows each\n")

    # Random by default, and per invocation. Fixed seeds would make the *second*
    # run of this script a cache measurement: identical bytes hash to the run key
    # the server already has, and it answers with the stored artifacts in about a
    # second. Impressive, and about nothing.
    seed_base = args.seed if args.seed is not None else random.randrange(2**31)
    print(f"seed        {seed_base} (pass --seed {seed_base} to regenerate these exact datasets)")
    datasets = [(index, synthetic_csv(args.rows, seed=seed_base + index)) for index in range(args.requests)]

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda item: upload(base, item[1], f"loadtest_{item[0]}.csv", args.api_key, args.timeout),
                datasets,
            )
        )
    wall = time.perf_counter() - wall_started

    accepted = [(seconds, job_id) for seconds, job_id, outcome in results if outcome == "accepted"]
    failures: dict[str, int] = {}
    for _, _, outcome in results:
        if outcome != "accepted":
            failures[outcome] = failures.get(outcome, 0) + 1

    upload_stats = summarise([seconds for seconds, _ in accepted])
    print("Upload accept latency")
    for key, value in upload_stats.items():
        print(f"  {key:<8} {value}")
    print(f"  accepted {len(accepted)}/{args.requests} in {wall:.1f}s ({len(accepted) / wall:.2f}/s)")
    if failures:
        print(f"  FAILED   {json.dumps(failures)}")

    # One run followed to the end. Under the queue this measures the worker;
    # inline, it measures the API doing the same work in the request path.
    training = {}
    if accepted:
        print("\nTraining one run to completion…")
        seconds, state = poll_until_done(base, accepted[0][1], args.api_key, args.train_timeout)
        training = {"seconds": round(seconds, 1), "state": state}
        print(f"  {state} in {seconds:.1f}s")

    after = readyz(base, args.api_key)
    depth = after["checks"]["queue"].get("depth")
    if depth is not None:
        print(f"  queue depth afterwards: {depth}")

    record = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "target": base,
        "environment": environment,
        "load": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "rows": args.rows,
            "seed": seed_base,
        },
        "upload_accept": upload_stats,
        "accepted": len(accepted),
        "failures": failures,
        "wall_seconds": round(wall, 1),
        "accepted_per_second": round(len(accepted) / wall, 2) if wall else None,
        "training": training,
        "queue_depth_after": depth,
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")
    print("\nPaste the numbers into docs/phase7-results.md — an unrecorded measurement is a rumour.")

    # A load test that could not upload is a failed run, not a slow one.
    return 0 if accepted and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
