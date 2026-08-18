import json
import logging
import time
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

import db
from auth import Principal, current_principal
from config import settings
from exceptions import DatasetTooLargeError, DatasetTooSmallError, PipelineError
from lifecycle import expiry_for_new_run
from logging_config import job_id_var
from ml.diagnostics import build_diagnostics
from ml.evaluate import per_class_report
from ml.explain import explain_pipeline
from ml.health import METRIC_LABEL, assess_run
from ml.model_card import build_model_card
from ml.preprocess import prepare_dataset
from ml.quality import data_quality_report, remove_duplicate_rows
from ml.train import HOLDOUT_SIZE, PIPELINE_VERSION, train_models
from ml.tuning import MAX_BUDGET_SECONDS
from observability import metrics
from utils.artifacts import dump_bundle
from utils.hashing import run_cache_key
from utils.helpers import (
    METADATA_DIR,
    MODEL_DIR,
    SPOOL_DIR,
    ensure_storage_dirs,
    now_utc_iso,
    read_csv_with_report,
    sanitize_text,
)
from utils.queue import get_queue
from utils.security import artifact_path
from utils.storage import ensure_local, publish_run
from utils.uploads import StoredUpload, read_upload_to_temp

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_chart_payload(df: pd.DataFrame) -> dict:
    charts = {"numeric_histograms": {}, "categorical_bars": {}}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()[:5]
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()[:5]
    for col in numeric_cols:
        bins = pd.cut(df[col], bins=10, include_lowest=True).value_counts().sort_index()
        charts["numeric_histograms"][col] = {
            "labels": [str(b) for b in bins.index.tolist()],
            "counts": [int(v) for v in bins.values.tolist()],
        }
    for col in categorical_cols:
        vc = df[col].astype(str).value_counts().head(10)
        charts["categorical_bars"][col] = {
            "labels": [sanitize_text(v, max_len=50) for v in vc.index.tolist()],
            "counts": [int(v) for v in vc.values.tolist()],
        }
    return charts


def _calculate_summary_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for col in df.columns:
        col_data = df[col]
        null_count = int(col_data.isnull().sum())
        if pd.api.types.is_numeric_dtype(col_data):
            stats[col] = {
                "type": "numeric",
                "mean": float(col_data.mean()) if not col_data.empty else 0,
                "median": float(col_data.median()) if not col_data.empty else 0,
                "min": float(col_data.min()) if not col_data.empty else 0,
                "max": float(col_data.max()) if not col_data.empty else 0,
                "std": float(col_data.std()) if not col_data.empty else 0,
                "null_count": null_count,
            }
        else:
            vc = col_data.astype(str).value_counts()
            top_val = vc.index[0] if not vc.empty else "N/A"
            stats[col] = {
                "type": "categorical",
                "most_frequent": str(top_val),
                "unique_count": int(col_data.nunique()),
                "null_count": null_count,
            }
    return stats


def _result_payload(metadata: dict, status: str) -> dict:
    """The shape the frontend consumes, built from stored metadata."""
    return {
        "status": status,
        "run_key": metadata.get("run_key"),
        "dataset_hash": metadata.get("dataset_hash"),
        "problem_type": metadata.get("problem_type"),
        "mode": metadata.get("mode"),
        "selected_model": metadata.get("selected_model"),
        "all_model_scores": metadata.get("all_model_scores", {}),
        # Phase 4: a score is a distribution plus one clean holdout number.
        "cv_scores": metadata.get("cv_scores", {}),
        "baseline_cv": metadata.get("baseline_cv", {}),
        "selection_metric": metadata.get("selection_metric"),
        "cv_folds": metadata.get("cv_folds"),
        "holdout_rows": metadata.get("holdout_rows"),
        "per_class": metadata.get("per_class", []),
        "imbalance": metadata.get("imbalance", {}),
        "class_weighting": metadata.get("class_weighting", {}),
        "smote_used": metadata.get("smote_used", False),
        "tuning": metadata.get("tuning", []),
        "tuning_budget_seconds": metadata.get("tuning_budget_seconds", 0.0),
        "ensemble_members": metadata.get("ensemble_members", []),
        "model_card": metadata.get("model_card", {}),
        "features": metadata.get("features", []),
        "dropped_columns": metadata.get("dropped_columns", []),
        "target": metadata.get("target"),
        "target_auto_detected": metadata.get("target_auto_detected"),
        "class_names": metadata.get("class_names", []),
        "failed_models": metadata.get("failed_models", {}),
        "health": metadata.get("health", {}),
        "explanation_method": metadata.get("explanation_method"),
        "feature_importance": metadata.get("feature_importance", []),
        "insights": metadata.get("insights", []),
        "summary_stats": metadata.get("summary_stats", {}),
        "quality_report": metadata.get("quality_report", {}),
        "charts": metadata.get("charts", {}),
        "diagnostics": metadata.get("diagnostics", {}),
        "parse_report": metadata.get("parse_report", {}),
        "row_count": metadata.get("row_count"),
        "column_count": metadata.get("column_count"),
        "timestamp": metadata.get("timestamp"),
    }


def _register_run(metadata: dict, filename: str | None, owner_id: str = db.LOCAL_OWNER_ID) -> None:
    """Upsert the registry row the history UI lists, from stored metadata."""
    quality = metadata.get("quality_report") or {}
    rows = metadata.get("row_count", quality.get("rows"))
    columns = metadata.get("column_count", quality.get("columns"))
    db.upsert_run(
        owner_id=owner_id,
        # Every artifact gets an owner and an expiry; the expiry is None when
        # retention is off, which is still an explicit answer.
        expires_at=expiry_for_new_run(),
        run_key=metadata["run_key"],
        dataset_hash=metadata.get("dataset_hash", ""),
        filename=filename,
        mode=metadata.get("mode", "auto"),
        target=metadata.get("target"),
        problem_type=metadata.get("problem_type"),
        selected_model=metadata.get("selected_model"),
        health_verdict=(metadata.get("health") or {}).get("verdict"),
        row_count=int(rows or 0),
        column_count=int(columns or 0),
        pipeline_version=metadata.get("pipeline_version"),
    )


def _run_upload_pipeline(
    job_id: str,
    stored: StoredUpload,
    mode: str,
    manual_model: str | None,
    target_column: str | None,
    filename: str | None = None,
    tuning_budget_seconds: float = 0.0,
    use_smote: bool = False,
    owner_id: str = db.LOCAL_OWNER_ID,
) -> None:
    # Every log line emitted below this point carries the job id.
    token = job_id_var.set(job_id)
    started = time.perf_counter()
    try:
        ensure_storage_dirs()
        mode = mode.lower()
        db.append_step(job_id, "Hashing dataset and run configuration", 5)
        # Digest was computed while streaming the upload to disk, so the file
        # is never hashed by pulling it back into memory.
        dataset_hash = stored.sha256
        file_bytes = stored.read_bytes()

        # Cached artifacts belong to a full configuration, not just a file:
        # the same CSV in Ensemble mode is a different run from Auto.
        # Scoped to the owner on purpose. Content-addressing means two users
        # uploading the same CSV would otherwise resolve to the same run key
        # and share artifacts — a cache hit for one would hand them a dataset
        # snapshot and model the other uploaded. Deduplication is not worth a
        # cross-tenant data leak, so the owner is part of the address.
        run_key = run_cache_key(
            dataset_hash,
            mode,
            manual_model,
            target_column,
            PIPELINE_VERSION,
            tuning_budget_seconds=tuning_budget_seconds,
            use_smote=use_smote,
            owner_id=owner_id,
        )
        logger.info("Pipeline started", extra={"run_key": run_key, "mode": mode, "target": target_column})

        metadata_path = artifact_path(METADATA_DIR, run_key, ".json")
        data_snapshot_path = artifact_path(METADATA_DIR, run_key, "_data.csv")
        model_path = artifact_path(MODEL_DIR, run_key, "_model.pkl")

        # With object storage the previous run may have been trained on another
        # machine, so "is it cached?" is a question about the bucket, not about
        # this container's disk. A no-op on the local backend.
        ensure_local(run_key)

        if metadata_path.exists() and model_path.exists():
            db.append_step(job_id, "Cache hit: loading artifacts for this exact configuration", 100)
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info("Cache hit", extra={"run_key": run_key})
            # A cache hit is still a completed run for this user. Registering it
            # here keeps the history consistent when artifacts outlive their
            # registry row — a fresh database, a restored volume, a deleted row.
            _register_run(metadata, filename, owner_id)
            metrics.observe_training("reused")
            db.complete_job(
                job_id,
                {
                    **_result_payload(metadata, status="reused"),
                    "message": "This dataset, mode, and target combination was already trained. Reused stored artifacts.",
                    "status_log": [item["step"] for item in (db.get_job(job_id) or {}).get("status_log", [])],
                },
                run_key,
            )
            return

        db.append_step(job_id, "Reading CSV and validating shape", 12)
        df, parse_report = read_csv_with_report(file_bytes)
        parse_warnings = parse_report.warnings
        for warn in parse_warnings:
            db.append_step(job_id, warn, 14)
        if df.shape[0] < 10 or df.shape[1] < 2:
            raise DatasetTooSmallError("Dataset too small. Need at least 10 rows and 2 columns.")
        if df.shape[0] > settings.max_upload_rows:
            raise DatasetTooLargeError(
                f"Dataset has {df.shape[0]:,} rows, above the {settings.max_upload_rows:,}-row limit."
            )

        db.append_step(job_id, "Running data quality checks", 20)
        quality = data_quality_report(df)
        df = remove_duplicate_rows(df)

        db.append_step(job_id, "Preparing auto-generated chart payload", 28)
        chart_payload = _build_chart_payload(df)

        db.append_step(job_id, "Selecting and validating target column", 35)
        prepared = prepare_dataset(df, target_column=target_column)
        target_col = prepared["target_col"]
        db.append_step(
            job_id,
            f"Target '{target_col}' accepted "
            f"({len(prepared['numeric_columns'])} numeric, {len(prepared['categorical_columns'])} categorical features)",
            45,
        )
        for warn in prepared["warnings"]:
            db.append_step(job_id, warn, 48)

        db.append_step(job_id, "Calculating summary statistics", 55)
        summary_stats = _calculate_summary_stats(df)

        db.append_step(job_id, "Holding out a test set, then cross-validating every candidate", 60)
        trained = train_models(
            prepared["X"],
            prepared["y"],
            preprocessor=prepared["preprocessor"],
            mode=mode,
            manual_model=manual_model,
            tuning_budget_seconds=tuning_budget_seconds,
            use_smote=use_smote,
        )
        problem_type = trained["problem_type"]
        trained_models = trained["trained_models"]
        scores = trained["performance_scores"]
        cv_scores = trained["cv_scores"]
        failed_models = trained.get("failed_models", {})
        label_encoder = trained["label_encoder"]
        imbalance = trained.get("imbalance", {})
        selection_metric = trained["selection_metric"]

        db.append_step(
            job_id,
            f"Cross-validated {len(cv_scores)} model(s) over {trained['cv_folds']} folds; "
            f"{trained['holdout_rows']:,} rows held out and never used for selection",
            76,
        )
        for warn in imbalance.get("warnings", []):
            db.append_step(job_id, warn, 77)
        if trained.get("smote_used"):
            db.append_step(job_id, "Applied SMOTE inside each fold's training rows only", 77)
        elif use_smote and trained.get("smote_note"):
            db.append_step(job_id, trained["smote_note"], 77)
        for report in trained.get("tuning", []):
            db.append_step(
                job_id,
                f"Tuned {report['model']} over {report['candidates']} candidates in {report['seconds']}s "
                f"({report['scoring']} {report['best_cv_score']:.4g})",
                78,
            )
        if failed_models:
            db.append_step(job_id, f"Some models failed and were skipped: {', '.join(failed_models.keys())}", 80)
            logger.warning("Models failed", extra={"failed": list(failed_models)})

        selected_model_name = trained["selected_model"]
        best_model_name = trained["explain_model"]
        selected_scores = scores[selected_model_name]
        selected_pipeline = trained_models[selected_model_name]
        # Attribution needs a preprocessor+model Pipeline, which a vote is not.
        explain_source = trained_models[best_model_name]
        selected_predictions = selected_pipeline.predict(trained["X_test"])

        db.append_step(
            job_id,
            f"Selected {selected_model_name} on cross-validated {METRIC_LABEL.get(selection_metric, selection_metric)}"
            + (f" — {trained['ensemble_note']}" if trained.get("ensemble_note") else ""),
            84,
        )

        db.append_step(job_id, "Computing feature attribution", 86)
        feature_importance, explain_method = explain_pipeline(
            explain_source, trained["X_train"], prepared["feature_columns"]
        )
        if explain_method == "native":
            db.append_step(job_id, "SHAP unavailable for this model; used the model's own feature importances", 87)
        elif explain_method == "unavailable":
            db.append_step(job_id, "No feature attribution available for this model", 87)

        db.append_step(job_id, "Assessing result health against a naive baseline", 90)
        health = assess_run(
            problem_type=problem_type,
            selected_scores=selected_scores,
            y_test=trained["y_test"],
            failed_models=failed_models,
            n_classes=len(trained["class_names"]) if trained["class_names"] else 0,
            cv_summary=cv_scores.get(selected_model_name, {}),
            imbalance=imbalance,
            holdout_rows=trained["holdout_rows"],
        )

        per_class = (
            per_class_report(trained["y_test"], selected_predictions, trained["class_names"])
            if problem_type == "classification"
            else []
        )

        db.append_step(job_id, "Building diagnostics (confusion/residuals, correlations)", 91)
        diagnostics = build_diagnostics(
            problem_type=problem_type,
            y_test=trained["y_test"],
            y_pred=selected_predictions,
            class_names=trained["class_names"],
            df=df,
        )

        db.append_step(job_id, "Generating insights", 92)
        cv_selected = cv_scores.get(selected_model_name, {}).get(selection_metric, {})
        insights = [
            f"Target column: '{target_col}'"
            + (" (auto-detected)." if prepared["target_auto_detected"] else " (chosen by you)."),
            f"Problem type detected as {problem_type}.",
            f"Training mode used: {mode}.",
            f"Models trained: {', '.join(trained_models.keys())}.",
            "Preprocessing was fitted on the training split only, so reported scores are leak-free.",
        ]
        metric_label = METRIC_LABEL.get(selection_metric, selection_metric)
        if cv_selected:
            insights.append(
                f"Selected on cross-validated {metric_label}: "
                f"{cv_selected['mean']:.4g} +/- {cv_selected['std']:.4g} over {trained['cv_folds']} folds, "
                "not on a single lucky split."
            )
            insights.append(
                f"{trained['holdout_rows']:,} rows were held out of cross-validation, tuning, and "
                "selection entirely, and scored once at the end."
            )
        if problem_type == "classification":
            insights.append(
                "Accuracy is reported but not selected on - macro F1 is, so a majority-class "
                "predictor cannot win on an imbalanced target."
            )
        insights.extend(imbalance.get("warnings", []))
        if trained.get("class_weighting"):
            insights.append(
                "Class weighting applied to: "
                + ", ".join(f"{name} ({how})" for name, how in trained["class_weighting"].items())
                + "."
            )
        if trained.get("smote_used"):
            insights.append("SMOTE oversampling was applied inside each fold's training rows only.")
        elif use_smote and trained.get("smote_note"):
            insights.append(trained["smote_note"])
        if trained.get("ensemble_members"):
            insights.append(
                "Ensemble members, weighted by how far each beat the baseline: "
                f"{', '.join(trained['ensemble_members'])}."
            )
        if trained.get("ensemble_note"):
            insights.append(trained["ensemble_note"])
        for report in trained.get("tuning", []):
            insights.append(
                f"Tuned {report['model']}: {report['best_params']} "
                f"({report['candidates']} candidates, {report['seconds']}s)."
            )
        if feature_importance:
            method_label = "SHAP" if explain_method == "shap" else "model-native"
            insights.append(
                f"Top influencing feature: {feature_importance[0]['feature']} ({method_label} attribution)."
            )
        insights.extend(prepared["warnings"])
        if failed_models:
            insights.append(f"Models that failed to train: {', '.join(failed_models.keys())}.")
        insights.extend([f"Parsing note: {w}" for w in parse_warnings])
        insights.extend([f"Quality warning: {w}" for w in quality["warnings"]])

        db.append_step(job_id, "Saving model bundle and metadata", 97)
        # One artifact: the fitted Pipeline already contains preprocessing, so
        # serving no longer has to re-apply a separately stored transformer.
        # An ensemble is now a fitted VotingClassifier/VotingRegressor - one
        # estimator like any other, which is why "kind" is always "single".
        bundle = {
            "kind": "single",
            "pipelines": {selected_model_name: selected_pipeline},
            "problem_type": problem_type,
            "label_encoder": label_encoder,
            "feature_columns": prepared["feature_columns"],
            "pipeline_version": PIPELINE_VERSION,
        }
        # Signed as it is written, so predict.py can prove this file is ours.
        dump_bundle(bundle, model_path)
        df.to_csv(data_snapshot_path, index=False)

        timestamp = now_utc_iso()
        model_card = build_model_card(
            run_key=run_key,
            timestamp=timestamp,
            dataset={
                "filename": filename,
                "dataset_hash": dataset_hash,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "mode": mode,
                "holdout_fraction": HOLDOUT_SIZE,
            },
            prepared=prepared,
            trained=trained,
            health=health,
            explanation_method=explain_method,
            feature_importance=feature_importance,
        )

        metadata = {
            "run_key": run_key,
            "dataset_hash": dataset_hash,
            "pipeline_version": PIPELINE_VERSION,
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "problem_type": problem_type,
            "mode": mode,
            "selected_model": selected_model_name,
            "all_model_scores": scores,
            "cv_scores": cv_scores,
            "baseline_cv": trained.get("baseline_cv", {}),
            "selection_metric": selection_metric,
            "cv_folds": trained["cv_folds"],
            "holdout_rows": trained["holdout_rows"],
            "per_class": per_class,
            "imbalance": imbalance,
            "class_weighting": trained.get("class_weighting", {}),
            "smote_used": trained.get("smote_used", False),
            "tuning": trained.get("tuning", []),
            "tuning_budget_seconds": trained.get("tuning_budget_seconds", 0.0),
            "ensemble_members": trained.get("ensemble_members", []),
            "model_card": model_card,
            "features": prepared["feature_columns"],
            "dropped_columns": prepared["dropped_columns"],
            "target": target_col,
            "target_auto_detected": prepared["target_auto_detected"],
            "class_names": [str(c) for c in trained["class_names"]],
            "failed_models": failed_models,
            "health": health,
            "explanation_method": explain_method,
            "timestamp": timestamp,
            "data_snapshot_path": str(data_snapshot_path),
            "quality_report": quality,
            "charts": chart_payload,
            "diagnostics": diagnostics,
            "parse_report": parse_report.as_dict(),
            "feature_importance": feature_importance,
            "insights": insights,
            "summary_stats": summary_stats,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Registry row: what the Phase 3 history UI lists. Built from the same
        # metadata a cache hit would read, so both paths register identically.
        _register_run(metadata, filename, owner_id)

        # Mirror the finished artifacts to durable storage, so another web
        # process can serve predictions for this run. Deliberately after the
        # registry row: a run that is listed but not yet mirrored degrades to a
        # cache miss, whereas mirroring first could publish a run nothing knows.
        published = publish_run(run_key)
        if published:
            logger.info("Artifacts published", extra={"run_key": run_key, "files": published})

        status_log = [item["step"] for item in (db.get_job(job_id) or {}).get("status_log", [])]
        db.complete_job(
            job_id,
            {
                **_result_payload(metadata, status="trained"),
                "trained_models": list(trained_models.keys()),
                "status_log": [*status_log, "Completed"],
            },
            run_key,
        )
        metrics.observe_training("trained", time.perf_counter() - started)
        logger.info(
            "Pipeline completed",
            extra={"run_key": run_key, "verdict": health.get("verdict"), "model": selected_model_name},
        )
    except PipelineError as exc:
        # Expected, user-fixable outcomes. No stack trace, and the UI gets a
        # kind it can word appropriately. Counted separately from a crash:
        # "the user sent a two-row CSV" is not an incident.
        logger.info("Pipeline rejected the input: %s", exc, extra={"kind": exc.kind})
        metrics.observe_training("rejected")
        db.fail_job(job_id, str(exc), exc.kind, f"Failed: {exc.kind}")
    except Exception as exc:
        logger.exception("Pipeline crashed")
        metrics.observe_training("failed")
        db.fail_job(job_id, str(exc), "pipeline", "Failed")
    finally:
        # The temp file exists only for the life of this job.
        stored.cleanup()
        job_id_var.reset(token)


@router.post("/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    manual_model: str | None = Form(None),
    target_column: str | None = Form(None),
    tuning_budget_seconds: float = Form(0.0),
    use_smote: bool = Form(False),
    principal: Principal = Depends(current_principal),
):
    # Streams to a temp file under a byte/row/type ceiling instead of reading
    # the whole upload into RAM. Raises 400/413 before any work is queued.
    stored = await read_upload_to_temp(file)
    target_column = (target_column or "").strip() or None
    # The budget is the user's to set, but not without a ceiling: this runs in
    # the web process, so an unbounded search would hold a worker indefinitely.
    tuning_budget_seconds = float(min(max(tuning_budget_seconds, 0.0), MAX_BUDGET_SECONDS))
    job_id = str(uuid4())
    db.create_job(job_id, "Upload received", owner_id=principal.user_id)

    queue = get_queue()
    if queue.out_of_process:
        # The worker is a different process — often a different container — so
        # the upload has to be handed over on shared storage rather than in a
        # private temp directory. The worker owns the file from here and
        # deletes it in the pipeline's finally block.
        ensure_storage_dirs()
        stored = stored.relocate(SPOOL_DIR)

    arguments = (
        job_id,
        stored,
        mode,
        manual_model,
        target_column,
        file.filename,
        tuning_budget_seconds,
        use_smote,
        principal.user_id,
    )
    logger.info(
        "Upload accepted",
        extra={
            "job": job_id,
            "mode": mode,
            "bytes": stored.size,
            "tuning_budget": tuning_budget_seconds,
            "queue": queue.name,
        },
    )

    if queue.out_of_process:
        # Named after the job id so a re-delivery cannot start the run twice.
        queue.enqueue(_run_upload_pipeline, *arguments, job_id=job_id)
    else:
        background_tasks.add_task(_run_upload_pipeline, *arguments)

    return {
        "job_id": job_id,
        "state": "queued",
        "queue": queue.name,
        "message": "Upload accepted. Poll /api/upload/status/{job_id} for live progress.",
    }


@router.get("/upload/status/{job_id}")
def upload_status(job_id: str, principal: Principal = Depends(current_principal)):
    from auth import owner_scope

    job = db.get_job(job_id, owner_id=owner_scope(principal))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
