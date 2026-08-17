import json
import logging
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from config import max_upload_rows
from ml.ensemble import ensemble_predict
from ml.evaluate import evaluate_classification, evaluate_regression
from ml.explain import explain_pipeline
from ml.health import assess_run
from ml.preprocess import TargetValidationError, prepare_dataset
from ml.quality import data_quality_report, remove_duplicate_rows
from ml.train import PIPELINE_VERSION, train_models
from utils.hashing import run_cache_key
from utils.helpers import METADATA_DIR, MODEL_DIR, ensure_storage_dirs, now_utc_iso, read_csv_flexible, sanitize_text
from utils.security import artifact_path
from utils.uploads import StoredUpload, read_upload_to_temp

logger = logging.getLogger(__name__)

router = APIRouter()
UPLOAD_JOBS: dict[str, dict] = {}


def _select_best_model(performance_scores: dict, problem_type: str) -> str:
    metric = "accuracy" if problem_type == "classification" else "rmse"
    reverse = problem_type == "classification"
    return sorted(performance_scores.keys(), key=lambda m: performance_scores[m][metric], reverse=reverse)[0]


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


def _update_job(job_id: str, step: str, progress: int) -> None:
    job = UPLOAD_JOBS[job_id]
    job["state"] = "running"
    job["current_step"] = step
    job["progress"] = progress
    job["status_log"].append({"step": step, "progress": progress, "timestamp": now_utc_iso()})


def _run_upload_pipeline(
    job_id: str,
    stored: StoredUpload,
    mode: str,
    manual_model: str | None,
    target_column: str | None,
) -> None:
    try:
        ensure_storage_dirs()
        mode = mode.lower()
        _update_job(job_id, "Hashing dataset and run configuration", 5)
        # Digest was computed while streaming the upload to disk, so the file
        # is never hashed by pulling it back into memory.
        dataset_hash = stored.sha256
        file_bytes = stored.read_bytes()

        # Cached artifacts belong to a full configuration, not just a file:
        # the same CSV in Ensemble mode is a different run from Auto.
        run_key = run_cache_key(dataset_hash, mode, manual_model, target_column, PIPELINE_VERSION)

        metadata_path = artifact_path(METADATA_DIR, run_key, ".json")
        data_snapshot_path = artifact_path(METADATA_DIR, run_key, "_data.csv")
        model_path = artifact_path(MODEL_DIR, run_key, "_model.pkl")

        if metadata_path.exists() and model_path.exists():
            _update_job(job_id, "Cache hit: loading artifacts for this exact configuration", 100)
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            UPLOAD_JOBS[job_id]["state"] = "completed"
            UPLOAD_JOBS[job_id]["progress"] = 100
            UPLOAD_JOBS[job_id]["result"] = {
                **_result_payload(metadata, status="reused"),
                "message": "This dataset, mode, and target combination was already trained. Reused stored artifacts.",
                "status_log": [item["step"] for item in UPLOAD_JOBS[job_id]["status_log"]],
            }
            return

        _update_job(job_id, "Reading CSV and validating shape", 12)
        df, parse_warnings = read_csv_flexible(file_bytes)
        for warn in parse_warnings:
            _update_job(job_id, warn, 14)
        if df.shape[0] < 10 or df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Dataset too small. Need at least 10 rows and 2 columns.")
        row_cap = max_upload_rows()
        if df.shape[0] > row_cap:
            raise HTTPException(
                status_code=413,
                detail=f"Dataset has {df.shape[0]:,} rows, above the {row_cap:,}-row limit.",
            )

        _update_job(job_id, "Running data quality checks", 20)
        quality = data_quality_report(df)
        df = remove_duplicate_rows(df)

        _update_job(job_id, "Preparing auto-generated chart payload", 28)
        chart_payload = _build_chart_payload(df)

        _update_job(job_id, "Selecting and validating target column", 35)
        prepared = prepare_dataset(df, target_column=target_column)
        target_col = prepared["target_col"]
        _update_job(
            job_id,
            f"Target '{target_col}' accepted "
            f"({len(prepared['numeric_columns'])} numeric, {len(prepared['categorical_columns'])} categorical features)",
            45,
        )
        for warn in prepared["warnings"]:
            _update_job(job_id, warn, 48)

        _update_job(job_id, "Calculating summary statistics", 55)
        summary_stats = _calculate_summary_stats(df)

        _update_job(job_id, "Splitting train/test, then fitting preprocessing on train only", 60)
        trained = train_models(
            prepared["X"],
            prepared["y"],
            preprocessor=prepared["preprocessor"],
            mode=mode,
            manual_model=manual_model,
        )
        problem_type = trained["problem_type"]
        trained_models = trained["trained_models"]
        scores = trained["performance_scores"]
        failed_models = trained.get("failed_models", {})
        label_encoder = trained["label_encoder"]

        _update_job(job_id, f"Trained: {', '.join(trained_models.keys())}", 78)
        if failed_models:
            _update_job(job_id, f"Some models failed and were skipped: {', '.join(failed_models.keys())}", 80)

        best_model_name = _select_best_model(scores, problem_type)
        if mode == "ensemble":
            selected_model_name = "Ensemble"
            # Score the vote itself on the held-out split. Reporting the best
            # member's score as the ensemble's would be the same kind of
            # flattering mislabel this phase exists to remove.
            ensemble_preds = ensemble_predict(trained_models, trained["X_test"], problem_type)
            scores["Ensemble"] = (
                evaluate_classification(trained["y_test"], ensemble_preds)
                if problem_type == "classification"
                else evaluate_regression(trained["y_test"], ensemble_preds)
            )
            selected_scores = scores["Ensemble"]
            explain_source = trained_models[best_model_name]
        else:
            selected_model_name = next(iter(trained_models)) if mode == "manual" else best_model_name
            selected_scores = scores[selected_model_name]
            explain_source = trained_models[selected_model_name]

        _update_job(job_id, "Computing feature attribution", 86)
        feature_importance, explain_method = explain_pipeline(
            explain_source, trained["X_train"], prepared["feature_columns"]
        )
        if explain_method == "native":
            _update_job(job_id, "SHAP unavailable for this model; used the model's own feature importances", 87)
        elif explain_method == "unavailable":
            _update_job(job_id, "No feature attribution available for this model", 87)

        _update_job(job_id, "Assessing result health against a naive baseline", 90)
        health = assess_run(
            problem_type=problem_type,
            selected_scores=selected_scores,
            y_test=trained["y_test"],
            failed_models=failed_models,
            n_classes=len(trained["class_names"]) if trained["class_names"] else 0,
        )

        _update_job(job_id, "Generating insights", 92)
        insights = [
            f"Target column: '{target_col}'"
            + (" (auto-detected)." if prepared["target_auto_detected"] else " (chosen by you)."),
            f"Problem type detected as {problem_type}.",
            f"Training mode used: {mode}.",
            f"Models trained: {', '.join(trained_models.keys())}.",
            "Preprocessing was fitted on the training split only, so reported scores are leak-free.",
        ]
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

        _update_job(job_id, "Saving model bundle and metadata", 97)
        # One artifact: the fitted Pipeline(s) already contain preprocessing, so
        # serving no longer has to re-apply a separately stored transformer.
        bundle = {
            "kind": "ensemble" if mode == "ensemble" else "single",
            "pipelines": trained_models if mode == "ensemble" else {selected_model_name: explain_source},
            "problem_type": problem_type,
            "label_encoder": label_encoder,
            "feature_columns": prepared["feature_columns"],
            "pipeline_version": PIPELINE_VERSION,
        }
        joblib.dump(bundle, model_path)
        df.to_csv(data_snapshot_path, index=False)

        metadata = {
            "run_key": run_key,
            "dataset_hash": dataset_hash,
            "pipeline_version": PIPELINE_VERSION,
            "problem_type": problem_type,
            "mode": mode,
            "selected_model": selected_model_name,
            "all_model_scores": scores,
            "features": prepared["feature_columns"],
            "dropped_columns": prepared["dropped_columns"],
            "target": target_col,
            "target_auto_detected": prepared["target_auto_detected"],
            "class_names": [str(c) for c in trained["class_names"]],
            "failed_models": failed_models,
            "health": health,
            "explanation_method": explain_method,
            "timestamp": now_utc_iso(),
            "data_snapshot_path": str(data_snapshot_path),
            "quality_report": quality,
            "charts": chart_payload,
            "feature_importance": feature_importance,
            "insights": insights,
            "summary_stats": summary_stats,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        UPLOAD_JOBS[job_id]["state"] = "completed"
        UPLOAD_JOBS[job_id]["progress"] = 100
        UPLOAD_JOBS[job_id]["current_step"] = "Completed"
        UPLOAD_JOBS[job_id]["status_log"].append({"step": "Completed", "progress": 100, "timestamp": now_utc_iso()})
        UPLOAD_JOBS[job_id]["result"] = {
            **_result_payload(metadata, status="trained"),
            "trained_models": list(trained_models.keys()),
            "status_log": [item["step"] for item in UPLOAD_JOBS[job_id]["status_log"]],
        }
    except TargetValidationError as exc:
        # A bad target is a user-fixable problem, not a crash; say so plainly.
        logger.info("Job %s rejected the target column: %s", job_id, exc)
        UPLOAD_JOBS[job_id]["state"] = "failed"
        UPLOAD_JOBS[job_id]["error"] = str(exc)
        UPLOAD_JOBS[job_id]["error_kind"] = "target"
        UPLOAD_JOBS[job_id]["current_step"] = "Failed: unusable target column"
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        UPLOAD_JOBS[job_id]["state"] = "failed"
        UPLOAD_JOBS[job_id]["error"] = getattr(exc, "detail", None) or str(exc)
        UPLOAD_JOBS[job_id]["error_kind"] = "pipeline"
        UPLOAD_JOBS[job_id]["current_step"] = "Failed"
    finally:
        # The temp file exists only for the life of this job.
        stored.cleanup()


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
    }


@router.post("/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    manual_model: str | None = Form(None),
    target_column: str | None = Form(None),
):
    # Streams to a temp file under a byte/row/type ceiling instead of reading
    # the whole upload into RAM. Raises 400/413 before any work is queued.
    stored = await read_upload_to_temp(file)
    target_column = (target_column or "").strip() or None
    job_id = str(uuid4())
    UPLOAD_JOBS[job_id] = {
        "job_id": job_id,
        "state": "queued",
        "current_step": "Queued",
        "progress": 0,
        "status_log": [{"step": "Upload received", "progress": 0, "timestamp": now_utc_iso()}],
        "result": None,
        "error": None,
    }
    background_tasks.add_task(_run_upload_pipeline, job_id, stored, mode, manual_model, target_column)
    return {
        "job_id": job_id,
        "state": "queued",
        "message": "Upload accepted. Poll /api/upload/status/{job_id} for live progress.",
    }


@router.get("/upload/status/{job_id}")
def upload_status(job_id: str):
    job = UPLOAD_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job
