# Keeping a model honest after it ships

Every phase before this one judges a model at the moment it is trained: the
holdout score, the health verdict, the model card. None of it is ever
revisited. A model that was defensible in August and quietly wrong by November
fails exactly the way this project was written about — confidently, with a
number, and with nothing saying otherwise.

This is the "afterwards": logging what a model is actually asked, watching
whether that has moved, saying honestly how far a reported confidence can be
trusted, and giving a retrained model a real argument to win before it
replaces the one being served.

Unlike Phase 6 and Phase 8, most of this is **on by default**. The difference
is deliberate: a rate limit changes what a caller may do, and a single-operator
install should not be limited by itself. Logging what a model already answered
changes nothing about behaviour — it only stops throwing that record away. A
model nobody is watching is the exact failure this phase exists to prevent, so
the safe default is the one that watches.

---

## What is on by default, and what it costs

```bash
PREDICTION_LOGGING_ENABLED=true     # off drops volume, latency, drift — everything
PREDICTION_LOG_INPUTS=true          # off keeps volume/latency, drops drift entirely
PREDICTION_LOG_MAX_ROWS=200         # a batch beyond this is sampled, not dropped
CALIBRATE_PROBABILITIES=true        # off leaves calibration measured, not corrected
CHALLENGER_PROMOTION_MARGIN=0.01    # how much better a retrain has to be
```

The cost of `PREDICTION_LOG_INPUTS=true` is the one to weigh deliberately:
feature values are stored, per prediction, for as long as the run's retention
allows. They are exactly as sensitive as the training rows they resemble —
disclosed at `GET /api/privacy`, owner-scoped, and deleted with the run. If
that trade is wrong for a deployment, turning inputs off keeps volume and
latency monitoring and gives up drift detection specifically; turning logging
off entirely gives up all of it, including "can this prediction be
reproduced".

---

## Prediction logging

Every served prediction — batch or single row — is written with its inputs,
its output, the model version that answered, and how long it took.

```bash
curl http://localhost:8000/api/runs/{run_key}/predictions
curl http://localhost:8000/api/predictions/{prediction_id}
```

A few things about the shape of it:

- **One row per predicted row, not per request.** A batch of 500 is 500 log
  entries tied together by `request_id`. Both drift and "how many predictions
  today" need per-row granularity; a request-level log would need to be
  re-expanded to get either.
- **A batch above `PREDICTION_LOG_MAX_ROWS` is sampled, evenly across the
  file, not truncated to its head.** A CSV is often sorted — by date, by
  customer id, by whatever the extract was ordered on — and keeping the first
  200 rows of a 50,000-row file would give drift a systematically
  unrepresentative sample with no way to notice. Every sampled row is flagged
  `sampled: true`.
- **The write is best-effort and never blocks the answer.** A caller gets
  their prediction whether or not the log write succeeds; a failure is logged
  loudly and the row is simply missing. The application already makes this
  trade for the audit log, and makes it the same way here — a gap in a chart
  is a cheaper failure than an outage on a prediction.

---

## Drift

The **reference** is the distribution each feature had at training time,
computed once and stored in the run's metadata. The **live** side is read back
from the prediction log. Comparing the two is the only measurement here that
needs no labels — you cannot know whether a prediction was *right* until an
outcome arrives, which may be never, but you can know the inputs no longer
look like the ones the model learned from.

```bash
curl http://localhost:8000/api/runs/{run_key}/drift
```

**The measure is Population Stability Index**, per feature, with the
conventional bands: below 0.10 is stable, 0.10–0.25 is moderate, above 0.25 is
material. PSI has no p-value — these are bands, not a significance test, and
nothing here should be read as one.

**Numeric columns are compared on quantile bins, not equal-width ones.** This
was not the first version. Equal-width bins on a skewed feature — spend,
income, almost anything measured in money — put most of the training mass in
one or two bins and leave the rest holding a handful of rows each, and PSI then
spends nearly all its signal on those thin bins. In this module's own
verification, that shape read a *same-distribution* batch of a few hundred rows
as material drift on nearly every trial. Quantile bins start with equal mass by
construction, which is what makes the number mean something at a real sample
size.

**A verdict needs `DRIFT_MIN_ROWS` predictions (200 by default) before it is
given at all.** Below the floor the endpoint reports `insufficient_data` and
says how many more are needed — not because nothing can be computed, but
because a PSI computed from a handful of rows is noise wearing a number. That
floor is also not a guess: it is the point where repeated simulation of a
same-distribution sample stopped producing false alarms, five-bin PSI and all
(`backend/tests/test_drift.py` has the simulation, and it re-runs on every test
pass rather than only being asserted in a comment).

**A category the training data never saw gets counted separately from PSI**,
because PSI over a fixed label set literally cannot represent a new one — it
is not one of the bins. A brand-new category above 5% of a batch escalates the
verdict past what its raw mass alone would score, because a new region code or
a renamed level is schema breakage wearing drift's clothes, and it is worth
noticing as such.

**A run trained before this phase has no stored reference**, and is not left
blind: the API falls back to the chart payloads already saved for the UI —
five numeric and five categorical columns, ten categories each — and says so
explicitly in the response (`derived_from: "charts"`), because a thin
reference must never look like a complete one. Re-train the run for full
coverage.

### What drift is not

Not an accuracy measurement. A feature can move a long way without the model
getting any worse, and a model can quietly rot while every input distribution
holds still. Drift buys a question asked early — it does not answer it. The
answer still needs labels, which is what retraining brings.

---

## Calibration

A confidence shown next to a prediction is a claim: "of the times I said 90%,
about 90% were right." Nothing checked that claim before this phase, and for
most classifiers it is false by construction — a forest's vote share is not a
probability, and a boosted tree tends to be overconfident near the extremes.

```bash
# in the monitoring view and in every prediction response
"calibration": {"verdict": "well_calibrated" | "usable" | "poor" | "unmeasured", ...}
```

At training time, a calibrated variant (`CalibratedClassifierCV`, isotonic or
sigmoid depending on how much development data there is) is fitted on the
development split and scored, alongside the original, on the untouched
holdout. **It replaces the served model only if it improves expected
calibration error by a real margin** — a marginal change on a holdout that
also decided *whether* to adopt it is as likely to be noise as improvement,
and adopting noise is a second layer of fitting for nothing. Both the before
and after numbers are kept in the run's metadata either way, so "we measured
and it was already fine" is as visible as "we measured and fixed it".

Below `MIN_HOLDOUT_ROWS` (50) the error is measured and reported but never
acted on — the same reasoning as the drift floor, at a different sample size
because ten calibration bins need less data to stabilise than ten stability-index
bins do across a joint feature comparison.

---

## Serving-time schema checks

The old message was `Missing required features: ['spend']`, correct and
nearly useless when the file plainly has a spend column, spelled differently.
Now:

```
1 required column(s) look renamed rather than absent — rename 'spend_usd' → 'spend' and try again.
```

Renames are detected first by normalising case, spaces, underscores and
dashes — the overwhelming majority of real ones — and only then by a
conservative fuzzy match, high enough to stay silent rather than guess wrong.

**A column that was numeric at training time and arrives unable to convert is
refused, not silently imputed.** The old path let a mostly-broken numeric
column reach the pipeline, which filled every unparseable cell with the
training median and returned a confident number for a value nobody supplied —
the exact silent-failure shape this whole project exists to prevent, moved
from training time to serving time. A couple of dirty cells in a large column
is still handled by the model's own imputation; a third of the column failing
to convert is refused, with an example of the value that would not parse.

---

## Retraining and promotion

```bash
curl -X POST http://localhost:8000/api/runs/{run_key}/retrain \
     -F "file=@newer_labelled_data.csv" \
     -F "trigger=manual"
```

A retrain fits a **challenger** with the run's existing configuration — same
target, same mode, same manual model if one was pinned — on the file supplied.
It does **not** replace the champion by default. It replaces it only when all
three of these hold:

- **the same metric** the run was originally selected on — choosing a new one
  at promotion time is choosing the metric that flatters the new model;
- **by a margin** (`CHALLENGER_PROMOTION_MARGIN`, 1% by default) — refit the
  same model on the same data with a different seed and the score moves on its
  own; promoting on any improvement at all means promoting that noise, forever;
- **on the same rows** — specifically the challenger's own holdout, the only
  labelled data neither model has been fitted on. The champion has never seen
  it; the challenger was not trained on it.

That last point has a known bias, stated rather than hidden: the challenger's
holdout is drawn from the *newer* data, so a champion trained on an older
distribution is being judged on the world as it now is. That is the right
question when retraining was triggered by drift — but it means the comparison
favours the challenger exactly when retraining gets triggered. The refusal
path is what keeps this honest: when the challenger does not clear the margin
even with that advantage, it is decisively not better, and the response says
so with both numbers rather than asserting it.

The response is the full comparison whether or not anything was promoted:

```json
{
  "promoted": true,
  "comparison": {
    "metric": "f1_macro", "champion": 0.7037, "challenger": 0.7186,
    "improvement": 0.0149, "required": 0.0070,
    "reason": "The challenger scored 0.7186 against the champion's 0.7037 ..."
  }
}
```

Every promotion — and every refusal — is written to the audit log
(`model.promoted` / `model.challenged`), because a promotion changes what
every caller of a run is served from that moment on, and that is exactly the
kind of act Phase 8's audit log exists for.

**Synchronous, not queued — a deliberate departure from the plan's suggestion
to reuse Phase 6's queue.** The response *is* the decision; a job id that has
to be polled to find out whether a model was replaced turns a decision into a
notification, and the whole point of the comparison is to hand back both
scores immediately. The training work inside it is bounded the same way any
other training is: it inherits the run's own `tuning_budget_seconds`, already
capped at upload time, and it is checked against Phase 8's per-account
concurrency ceiling before it starts even though it writes no row to `jobs` —
see `routes/monitoring.py`. What this does not get is the queue's actual
benefit: a very large or slow retrain still blocks the request thread it runs
on, rather than handing off to a separate worker. That is the trade to
revisit first if retraining stops being an occasional, deliberate action and
starts being routine at a size where that matters.

**No automatic promotion on a schedule, and no retraining triggered by drift
without a person.** `GET /api/monitoring/due` reports runs whose champion is
old enough that a scheduled retrain is reasonable — mirroring the retention
purge's own reasoning: an in-process scheduler is one more thing that dies with
the process, so this reports what is due and a cron entry calls the endpoint:

```cron
0 3 * * 0  curl -s -X POST http://localhost:8000/api/runs/$RUN_KEY/retrain \
                -F "file=@/data/latest.csv" -F "trigger=scheduled" \
                -H "Authorization: Bearer $API_KEY"
```

A drift signal is a good reason to *look* — `overview.retraining.recommended`
says so, with the reason and the endpoint to call. It is a bad reason to fit a
model on whatever anomaly produced the signal and serve it without a person
deciding to.

---

## The monitoring view

```bash
curl http://localhost:8000/api/runs/{run_key}/monitoring
```

One call: prediction volume and latency percentiles, drift status, calibration,
the retraining advice, the full version history with its champion, and the
model card — which existed since Phase 5 and had nowhere to live once training
day ended. `GET /api/runs/{run_key}/versions` on its own is the version history
without the rest, for a page that only needs that.

---

## Deletion and retention

Predictions and model versions are deleted with the run that owns them —
`DELETE /api/runs/{run_key}` and `DELETE /api/account/data` both purge them, and
so does the retention policy once a run expires. This is the deliberate
opposite of the audit log, which is kept *outside* retention because it has to
outlive what it records. A prediction log describes a model that will not.

---

## What this phase does not do

- **No fairness or subgroup monitoring.** Drift and calibration are both
  computed in aggregate; neither says whether a specific group is served worse
  than another.
- **No outcome-based accuracy tracking after serving.** Everything here is
  either input-side (drift, schema) or measured once at training/retraining
  time (calibration, the champion/challenger comparison). Closing the loop with
  real-world outcomes needs a label to arrive after the fact, and nothing
  collects one.
- **The champion/challenger comparison is a single holdout split**, not
  repeated cross-validation. A margin absorbs some of that split's sampling
  noise; it does not eliminate it.
- **PSI is univariate.** Each feature is compared on its own; a joint shift
  that leaves every marginal distribution looking stable is invisible to it.
