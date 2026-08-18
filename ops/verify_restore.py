"""Prove a restored stack actually works.

    python ops/verify_restore.py [--base-url http://localhost:8000] [--api-key ...]

`pg_restore` exiting zero says the rows arrived. It says nothing about whether
a run from before the restore still opens, or whether its model still loads —
and the model is the part most likely to be quietly broken, because a bundle
whose signature cannot be verified is refused rather than loaded, and a stack
restored without its original ARTIFACT_SIGNING_KEY has exactly that problem.

So this checks the whole path end to end, on a run that existed before:

    /readyz -> the schema is at head, and what degraded is visible
    /api/runs -> the registry survived
    /api/runs/{key}/result -> the metadata artifact survived
    /api/predict/{key}/row -> the model bundle survived *and* verified

Standard library only, so it runs against a deployment from anywhere, including
a container with nothing installed in it.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

TIMEOUT = 30


class Failure(Exception):
    """Something the restore was supposed to preserve did not survive."""


def request(url: str, api_key: str | None = None, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=TIMEOUT) as reply:
            return json.loads(reply.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise Failure(f"{url} answered {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise Failure(f"{url} is not reachable: {exc.reason}") from exc


def sample_row(features: list[str]) -> dict:
    """One row of filler, one value per feature the run was trained on.

    The value does not matter and is not supposed to: this asks whether the
    stored pipeline loads, verifies, and runs — not whether its answer is any
    good. The preprocessing in the bundle imputes and encodes whatever it is
    given, including a category it has never seen.
    """
    return dict.fromkeys(features, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None, help="required when AUTH_ENABLED is true")
    parser.add_argument("--run-key", default=None, help="check this run instead of the most recent one")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print(f"Verifying {base}")

    ready = request(f"{base}/readyz", args.api_key)
    schema = ready["checks"]["schema"]
    if not schema["ok"]:
        raise Failure(f"schema is at {schema.get('revision')}, code expects {schema.get('head')}")
    print(f"  readyz          ok   schema {schema['revision']}, storage {ready['checks']['storage']['backend']}")
    if not ready["checks"]["artifacts"]["ok"]:
        # Not fatal here, but it is the most likely cause of the failure two
        # steps below, so say it before that happens rather than after.
        print(f"  artifacts       WARN {ready['checks']['artifacts']['detail']}")

    listing = request(f"{base}/api/runs", args.api_key)
    if not listing["runs"]:
        raise Failure("the restored database has no runs in it — nothing was preserved, or auth is scoping them away")
    run = next((r for r in listing["runs"] if r["run_key"] == args.run_key), listing["runs"][0])
    run_key = run["run_key"]
    print(f"  runs            ok   {listing['count']} run(s); checking {run_key[:12]}… ({run.get('filename')})")

    result = request(f"{base}/api/runs/{run_key}/result", args.api_key)
    features = result.get("features") or []
    if not features:
        raise Failure(f"{run_key} has no feature list — its metadata artifact did not survive the restore")
    print(f"  metadata        ok   {len(features)} feature(s), target {result.get('target')!r}")

    prediction = request(
        f"{base}/api/predict/{run_key}/row",
        args.api_key,
        payload={"row": sample_row(features)},
    )
    if not prediction.get("predictions"):
        raise Failure(f"{run_key} answered without a prediction: {json.dumps(prediction)[:300]}")
    print(
        f"  prediction      ok   {prediction['predictions'][0]!r} — the stored bundle loaded and its signature verified"
    )

    print("\nPASS — a run that existed before the restore still opens and still predicts.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failure as failure:
        print(f"\nFAIL — {failure}", file=sys.stderr)
        sys.exit(1)
