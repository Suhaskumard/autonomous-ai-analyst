import json
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from ml.explain import explain_model
from ml.preprocess import preprocess_dataset
from ml.quality import data_quality_report, remove_duplicate_rows
from ml.train import train_models
from utils.hashing import dataset_sha256
from utils.helpers import METADATA_DIR, MODEL_DIR, ensure_storage_dirs, now_utc_iso, read_csv_flexible, sanitize_text

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


def _run_upload_pipeline(job_id: str, file_bytes: bytes, mode: str, manual_model: str | None) -> None:
    try:
        ensure_storage_dirs()
        mode = mode.lower()
        _update_job(job_id, "Hashing dataset", 5)
        dataset_hash = dataset_sha256(file_bytes)

        metadata_path = METADATA_DIR / f"{dataset_hash}.json"
        data_snapshot_path = METADATA_DIR / f"{dataset_hash}_data.csv"
        model_path = MODEL_DIR / f"{dataset_hash}_model.pkl"
        pipeline_path = MODEL_DIR / f"{dataset_hash}_pipeline.pkl"

        if metadata_path.exists() and model_path.exists() and pipeline_path.exists():
            _update_job(job_id, "Cache hit: loading pre-trained artifacts", 100)
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            UPLOAD_JOBS[job_id]["state"] = "completed"
            UPLOAD_JOBS[job_id]["result"] = {
                "status": "reused",
                "dataset_hash": dataset_hash,
                "message": "Dataset already seen. Reused stored model and pipeline.",
                "mode": metadata.get("mode"),
                "selected_model": metadata.get("selected_model"),
                "problem_type": metadata.get("problem_type"),
                "all_model_scores": metadata.get("all_model_scores", {}),
                "features": metadata.get("features", []),
                "target": metadata.get("target"),
                "feature_importance": metadata.get("feature_importance", []),
                "insights": metadata.get("insights", []),
                "quality_report": metadata.get("quality_report", {}),
                "charts": metadata.get("charts", {}),
                "status_log": [item["step"] for item in UPLOAD_JOBS[job_id]["status_log"]],
            }
            return

        _update_job(job_id, "Reading CSV and validating shape", 12)
        df, parse_warnings = read_csv_flexible(file_bytes)
        for warn in parse_warnings:
            _update_job(job_id, warn, 14)
        if df.shape[0] < 10 or df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="Dataset too small. Need at least 10 rows and 2 columns.")

        _update_job(job_id, "Running data quality checks", 20)
        quality = data_quality_report(df)
        df = remove_duplicate_rows(df)

        _update_job(job_id, "Preparing auto-generated chart payload", 28)
        chart_payload = _build_chart_payload(df)

        _update_job(job_id, "Preprocessing: target detection", 35)
        processed = preprocess_dataset(df)
        _update_job(
            job_id,
            f"Preprocessing complete ({len(processed['numeric_columns'])} numeric, {len(processed['categorical_columns'])} categorical columns)",
            50,
        )

        _update_job(job_id, "Calculating summary statistics", 55)
        summary_stats = _calculate_summary_stats(df)
        if processed["dropped_target_rows"] > 0:
            _update_job(job_id, f"Dropped {processed['dropped_target_rows']} rows with missing target values", 54)

        _update_job(job_id, "Training models", 60)
        trained = train_models(processed["X_processed"], processed["y"], mode=mode, manual_model=manual_model)
        problem_type = trained["problem_type"]
        trained_models = trained["trained_models"]
        scores = trained["performance_scores"]
        failed_models = trained.get("failed_models", {})
        _update_job(job_id, f"Trained: {', '.join(trained_models.keys())}", 78)
        if failed_models:
            _update_job(job_id, f"Some models skipped due to errors: {', '.join(failed_models.keys())}", 80)

        if mode == "ensemble":
            selected_model_name = "Ensemble"
            selected_model = {"mode": "ensemble", "models": trained_models, "problem_type": problem_type}
            feature_source = trained_models[_select_best_model(scores, problem_type)]
        else:
            selected_model_name = next(iter(trained_models)) if mode == "manual" else _select_best_model(scores, problem_type)
            selected_model = trained_models[selected_model_name]
            feature_source = selected_model

        _update_job(job_id, "Computing SHAP explainability", 86)
        feature_importance = explain_model(feature_source, trained["X_train"], processed["feature_columns"])

        _update_job(job_id, "Generating insights", 92)
        insights = [
            f"Target column automatically detected as '{processed['target_col']}'.",
            f"Problem type detected as {problem_type}.",
            f"Training mode used: {mode}.",
            f"Models trained: {', '.join(trained_models.keys())}.",
        ]
        if processed["dropped_target_rows"] > 0:
            insights.append(f"Removed {processed['dropped_target_rows']} rows because target value was missing.")
        if feature_importance:
            insights.append(f"Top influencing feature: {feature_importance[0]['feature']}.")
        if failed_models:
            insights.append(f"Skipped models due to training errors: {', '.join(failed_models.keys())}.")
        insights.extend([f"Parsing note: {w}" for w in parse_warnings])
        insights.extend([f"Quality warning: {w}" for w in quality["warnings"]])

        _update_job(job_id, "Saving model, pipeline, and metadata", 97)
        joblib.dump(selected_model, model_path)
        joblib.dump(processed["preprocessing_pipeline"], pipeline_path)
        df.to_csv(data_snapshot_path, index=False)

        metadata = {
            "dataset_hash": dataset_hash,
            "problem_type": problem_type,
            "mode": mode,
            "selected_model": selected_model_name,
            "all_model_scores": scores,
            "features": processed["feature_columns"],
            "target": processed["target_col"],
            "failed_models": failed_models,
            "timestamp": now_utc_iso(),
            "data_snapshot_path": str(data_snapshot_path),
            "quality_report": quality,
            "charts": chart_payload,
            "feature_importance": feature_importance,
            "insights": insights,
            "summary_stats": summary_stats,
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        UPLOAD_JOBS[job_id]["state"] = "completed"
        UPLOAD_JOBS[job_id]["progress"] = 100
        UPLOAD_JOBS[job_id]["current_step"] = "Completed"
        UPLOAD_JOBS[job_id]["status_log"].append({"step": "Completed", "progress": 100, "timestamp": now_utc_iso()})
        UPLOAD_JOBS[job_id]["result"] = {
            "status": "trained",
            "dataset_hash": dataset_hash,
            "problem_type": problem_type,
            "mode": mode,
            "selected_model": selected_model_name,
            "trained_models": list(trained_models.keys()),
            "all_model_scores": scores,
            "features": processed["feature_columns"],
            "target": processed["target_col"],
            "failed_models": failed_models,
            "feature_importance": feature_importance,
            "insights": insights,
            "summary_stats": summary_stats,
            "quality_report": quality,
            "charts": chart_payload,
            "status_log": [item["step"] for item in UPLOAD_JOBS[job_id]["status_log"]],
        }
    except Exception as exc:
        UPLOAD_JOBS[job_id]["state"] = "failed"
        UPLOAD_JOBS[job_id]["error"] = str(exc)
        UPLOAD_JOBS[job_id]["current_step"] = "Failed"


@router.post("/upload")
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    manual_model: str | None = Form(None),
):
    file_bytes = await file.read()
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
    background_tasks.add_task(_run_upload_pipeline, job_id, file_bytes, mode, manual_model)
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
