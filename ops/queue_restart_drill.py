"""Restart the API in the middle of a training run and watch the run finish.

    docker compose --profile queue up -d
    python ops/queue_restart_drill.py

This is Phase 6's central promise and the one thing nothing had ever observed:
that a job survives the process that accepted it. Unit tests prove the handover
— the job goes to the queue, its input lands somewhere another process can read
it, the arguments pickle. None of that proves a worker picks it up, or that the
API comes back and reports the result of a job it did not run.

The drill:

    1. upload a dataset and wait until training has actually started;
    2. `docker compose restart backend` — the process that accepted the job;
    3. poll the status endpoint until the job is terminal.

It fails loudly if the queue is not really the queue. An `inline` backend runs
training inside the API, so restarting it kills the run and the job is marked
`interrupted` — which is the correct behaviour there and a failed drill here.

    --dataset PATH   use your own CSV instead of the checked-in fixture
    --service NAME   restart something else (try `worker`: RQ requeues the job)
"""

import argparse
import http.client
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "backend" / "tests" / "fixtures" / "clean_classification.csv"
POLL_SECONDS = 1.0


class DrillFailed(Exception):
    pass


def get_json(url: str, api_key: str | None) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except (OSError, http.client.HTTPException) as exc:
        # Deliberately broad. The whole point of this drill is to kill the API
        # mid-request, and a socket dying part-way through a response does not
        # raise URLError — it raises ConnectionAbortedError or RemoteDisconnected
        # straight from http.client. Catching only URLError meant the drill
        # crashed on the exact event it exists to produce.
        raise DrillFailed(f"{url}: {exc}") from exc


def upload(base: str, dataset: Path, api_key: str | None) -> str:
    """Multipart by hand, so this needs nothing that is not in the stdlib."""
    boundary = f"----analyst{uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{dataset.name}"\r\n'.encode(),
            b"Content-Type: text/csv\r\n\r\n",
            dataset.read_bytes(),
            f"\r\n--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="mode"\r\n\r\nauto\r\n',
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{base}/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=120) as reply:
        payload = json.loads(reply.read().decode("utf-8"))

    if payload.get("queue") != "rq":
        raise DrillFailed(
            f"The API accepted this job on the '{payload.get('queue')}' backend. Restarting it would "
            "kill the run, which is correct for inline execution and means this drill has nothing to "
            "prove. Start the queue profile and set QUEUE_BACKEND=rq."
        )
    return payload["job_id"]


def wait_for_progress(base: str, job_id: str, api_key: str | None, timeout: float) -> dict:
    """Wait until the worker has visibly started, not merely accepted, the job."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = get_json(f"{base}/api/upload/status/{job_id}", api_key)
        if status["state"] == "failed":
            raise DrillFailed(f"the job failed before the restart: {status.get('error')}")
        if status["state"] == "completed":
            if (status.get("result") or {}).get("status") == "reused":
                raise DrillFailed(
                    "this dataset has been trained on this stack before, so the upload was answered "
                    "from the content-addressed cache in about a second and no training happened at "
                    "all. Use a dataset the server has not seen — including on a previous attempt at "
                    "this drill."
                )
            raise DrillFailed(
                "the job finished before the restart — it is too small to prove anything. "
                "Use --dataset with something that takes longer to train."
            )
        if status["state"] == "running" and status["progress"] > 0:
            return status
        time.sleep(POLL_SECONDS)
    raise DrillFailed("no worker picked the job up — is the worker container running?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--service", default="backend", help="compose service to restart (default: backend)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for the job to finish")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    if not args.dataset.exists():
        raise DrillFailed(f"no such dataset: {args.dataset}")

    ready = get_json(f"{base}/readyz", args.api_key)
    print(f"queue={ready['checks']['queue']} storage={ready['checks']['storage']['backend']}")
    if not ready["checks"]["artifacts"]["ok"]:
        print(f"WARNING: {ready['checks']['artifacts']['detail']}")

    job_id = upload(base, args.dataset, args.api_key)
    print(f"job {job_id} accepted and handed to the queue")

    started = wait_for_progress(base, job_id, args.api_key, timeout=180)
    print(f"worker is running it: {started['progress']}% — {started['current_step']}")

    print(f"restarting the '{args.service}' container mid-training…")
    restarted_at = time.monotonic()
    subprocess.run(["docker", "compose", "restart", args.service], cwd=REPO_ROOT, check=True)
    print(f"restarted after {time.monotonic() - restarted_at:.1f}s; polling for the result")

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            status = get_json(f"{base}/api/upload/status/{job_id}", args.api_key)
        except DrillFailed:
            # Expected for a few seconds: the API is the thing being restarted.
            time.sleep(POLL_SECONDS)
            continue

        if status["state"] == "completed":
            print(f"\nPASS — the run finished at {status['progress']}%, run key {status['run_key'][:12]}…")
            print("The job outlived the process that accepted it, and the status endpoint reports it.")
            return 0
        if status["state"] == "failed":
            raise DrillFailed(
                f"the job is {status['state']} ({status.get('error_kind')}): {status.get('error')}\n"
                "'interrupted' here means the restart killed the run, which is what the queue exists to prevent."
            )
        time.sleep(POLL_SECONDS)

    raise DrillFailed(f"the job was still {status['state']} after {args.timeout}s")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DrillFailed as failure:
        print(f"\nFAIL — {failure}", file=sys.stderr)
        sys.exit(1)
