"""Exporting a trained pipeline so it can leave this application — Phase 10.

Two formats, and every run gets at least one of them.

**ONNX**, where the preprocessing and model allow it. `skl2onnx` converts the
sklearn half (`ColumnTransformer`, `SimpleImputer`, `StandardScaler`,
`OneHotEncoder`, and most estimators); `onnxmltools` registers converters for
XGBoost/LightGBM on top of it. Two things used elsewhere in this pipeline have
no ONNX converter at all, and this module refuses rather than silently
producing a graph that scores differently from the model it claims to be:

* `sklearn.preprocessing.TargetEncoder` (`ml/preprocess.py`, used for every
  high-cardinality categorical column) — `skl2onnx` has never had a converter
  for it;
* `HistGradientBoostingClassifier`/`Regressor` (`ml/train.py`'s registry) —
  historically unsupported for the same reason (no exposed converter).

**A portable bundle**, always available: the signed pickle this application
already trusts itself, a standalone scoring script, and a `requirements.txt`
pinned to the versions this server actually used to fit it — so a model can
run somewhere this application is not installed, without an ONNX conversion
having to succeed first.

Neither format needs an ensemble special case: since Phase 4, `bundle["kind"]`
is always `"single"` — an ensemble is a fitted `VotingClassifier`/
`VotingRegressor`, one estimator like any other (`routes/upload.py`).
"""

import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline as SklearnPipeline

logger = logging.getLogger(__name__)

try:
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_regressor_output_shapes

    SKL2ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the install
    SKL2ONNX_AVAILABLE = False
    logger.info("skl2onnx is not installed; ONNX export is unavailable.")

#: Estimators skl2onnx has no converter for, checked by class name so this
#: module does not have to import sklearn's HistGradientBoosting* just to name
#: them, matching how train.py's own optional imports are guarded.
_UNSUPPORTED_ESTIMATOR_NAMES = {"HistGradientBoostingClassifier", "HistGradientBoostingRegressor"}

_BOOSTED_TREE_CONVERTERS_REGISTERED = False


def _register_boosted_tree_converters() -> None:
    """Wire onnxmltools' XGBoost/LightGBM converters into skl2onnx's registry.

    Lazy and idempotent: it imports xgboost, lightgbm and onnxmltools, all
    optional, and this module has to import cleanly whether or not any of
    them are present.
    """
    global _BOOSTED_TREE_CONVERTERS_REGISTERED
    if _BOOSTED_TREE_CONVERTERS_REGISTERED or not SKL2ONNX_AVAILABLE:
        return
    _BOOSTED_TREE_CONVERTERS_REGISTERED = True

    try:
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from onnxmltools.convert.xgboost.shape_calculators.Classifier import (
            calculate_xgboost_classifier_output_shapes,
        )
        from xgboost import XGBClassifier, XGBRegressor

        update_registered_converter(
            XGBClassifier,
            "XGBoostXGBClassifier",
            calculate_xgboost_classifier_output_shapes,
            convert_xgboost,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        update_registered_converter(
            XGBRegressor, "XGBoostXGBRegressor", calculate_linear_regressor_output_shapes, convert_xgboost
        )
    except ImportError:
        logger.info("xgboost/onnxmltools XGBoost converters unavailable; XGBoost runs cannot export to ONNX.")

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
        from onnxmltools.convert.lightgbm.shape_calculators.Classifier import (
            calculate_lightgbm_classifier_output_shapes,
        )

        update_registered_converter(
            LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_lightgbm_classifier_output_shapes,
            convert_lightgbm,
            options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        )
        update_registered_converter(
            LGBMRegressor, "LightGbmLGBMRegressor", calculate_linear_regressor_output_shapes, convert_lightgbm
        )
    except ImportError:
        logger.info("lightgbm/onnxmltools LightGBM converters unavailable; LightGBM runs cannot export to ONNX.")


def _unwrap_resampling_pipeline(pipeline):
    """Drop an imblearn resampler step, if the pipeline has one.

    `ml/imbalance.py`'s `build_pipeline` wraps (preprocessor -> SMOTE ->
    model) in imblearn's Pipeline when SMOTE is on. imblearn's own
    `.predict()` already skips the resampler at inference — that is the
    whole point of it living inside the pipeline rather than being applied to
    the training frame directly — so a plain sklearn Pipeline over just
    (preprocessor -> model) scores identically to what serving already does,
    and it is a type skl2onnx has a converter for, where imblearn's Pipeline
    is not.
    """
    steps = list(getattr(pipeline, "steps", []) or [])
    if not steps:
        return pipeline
    is_imblearn = type(pipeline).__module__.split(".")[0] == "imblearn"
    kept = [(name, step) for name, step in steps if type(step).__module__.split(".")[0] != "imblearn"]
    if not is_imblearn and len(kept) == len(steps):
        return pipeline
    return SklearnPipeline(steps=kept)


def _pipeline_parts(bundle: dict) -> tuple[Any, Any, Any]:
    """(pipeline, preprocessor, model) from this bundle, resampler dropped."""
    pipeline = next(iter(bundle["pipelines"].values()))
    unwrapped = _unwrap_resampling_pipeline(pipeline)
    steps = dict(getattr(unwrapped, "steps", []))
    return unwrapped, steps.get("preprocessor"), steps.get("model")


def _target_encoded_columns(preprocessor) -> list[str]:
    """Columns run through sklearn's TargetEncoder, if any — the ONNX blocker."""
    from sklearn.preprocessing import TargetEncoder

    found: list[str] = []
    for _, transformer, columns in getattr(preprocessor, "transformers_", []):
        candidates = (
            [s for _, s in getattr(transformer, "steps", [])] if hasattr(transformer, "steps") else [transformer]
        )
        if any(isinstance(step, TargetEncoder) for step in candidates):
            found.extend(columns if isinstance(columns, list) else [columns])
    return found


def _column_kinds(preprocessor) -> tuple[list[str], list[str]]:
    """(numeric columns, categorical/one-hot columns) from a fitted ColumnTransformer."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for _, transformer, columns in preprocessor.transformers_:
        if isinstance(transformer, str):  # "drop" / "passthrough"
            continue
        steps = getattr(transformer, "steps", [])
        is_numeric = any(type(step).__name__ == "StandardScaler" for _, step in steps)
        target = numeric_cols if is_numeric else categorical_cols
        target.extend(columns if isinstance(columns, list) else [columns])
    return numeric_cols, categorical_cols


def onnx_availability(bundle: dict) -> dict:
    """Whether this run's pipeline can become ONNX, and why not when it cannot."""
    if not SKL2ONNX_AVAILABLE:
        return {"available": False, "reason": "skl2onnx is not installed on this server."}

    _, preprocessor, model = _pipeline_parts(bundle)
    if preprocessor is None or model is None:
        return {"available": False, "reason": "This run's pipeline has an unrecognised shape."}

    target_encoded = _target_encoded_columns(preprocessor)
    if target_encoded:
        return {
            "available": False,
            "reason": (
                f"This run target-encodes high-cardinality column(s) ({', '.join(target_encoded[:5])}) with "
                "sklearn's TargetEncoder, which has no ONNX converter. Use the portable bundle format instead."
            ),
        }

    # skl2onnx's SimpleImputer converter refuses a string-typed column unless
    # the fitted `missing_values` is itself a string (see imputer_op.py) —
    # `ml/preprocess.py`'s categorical branch always fits one with the
    # (float) default, `np.nan`, whether or not the column actually has any
    # missing values, because the imputer step is structural, not conditional
    # on the data. That makes any categorical/one-hot column an ONNX blocker
    # today, not just target-encoded ones — checked structurally here rather
    # than by attempting the conversion and parsing a library exception.
    _, categorical_cols = _column_kinds(preprocessor)
    if categorical_cols:
        return {
            "available": False,
            "reason": (
                f"This run has categorical feature(s) ({', '.join(categorical_cols[:5])}). skl2onnx's "
                "SimpleImputer converter does not support the string columns this pipeline's one-hot "
                "branch produces. ONNX export currently works only for numeric-only runs; use the "
                "portable bundle format instead."
            ),
        }

    model_name = type(model).__name__
    if model_name in _UNSUPPORTED_ESTIMATOR_NAMES:
        return {
            "available": False,
            "reason": f"{model_name} has no ONNX converter in skl2onnx. Use the portable bundle format instead.",
        }

    # Past this point nothing structural is known to be wrong, but skl2onnx
    # and onnxmltools converters have failure modes that are cheaper to
    # observe than to predict — XGBoost/LightGBM composed into a
    # skl2onnx-driven graph is a real example (a type-identity mismatch
    # between the two libraries' otherwise-equivalent tensor type classes).
    # A trial conversion is fast for a model this size, and an accurate
    # "no" here is worth more than an optimistic one a download then
    # contradicts.
    try:
        _convert(bundle)
    except Exception as exc:
        logger.info("ONNX conversion probe failed: %s", exc)
        return {
            "available": False,
            "reason": f"This pipeline could not be converted to ONNX ({exc}). Use the portable bundle format instead.",
        }

    return {"available": True, "reason": None}


def _convert(bundle: dict):
    """The actual skl2onnx conversion. Raises whatever skl2onnx/onnxmltools raise."""
    _register_boosted_tree_converters()
    unwrapped, preprocessor, _ = _pipeline_parts(bundle)
    numeric_cols, categorical_cols = _column_kinds(preprocessor)
    initial_types = [(col, FloatTensorType([None, 1])) for col in numeric_cols]
    initial_types += [(col, StringTensorType([None, 1])) for col in categorical_cols]
    return convert_sklearn(unwrapped, initial_types=initial_types)


def export_onnx(bundle: dict) -> bytes:
    """Convert this run's pipeline to ONNX. Raises ValueError if it cannot.

    Re-runs `onnx_availability`'s checks rather than trusting a caller already
    called it: the structural checks give a far more specific message than
    whatever exception a doomed conversion attempt would raise, and this is
    the one place that message actually reaches an API response.
    """
    availability = onnx_availability(bundle)
    if not availability["available"]:
        raise ValueError(availability["reason"])

    try:
        onnx_model = _convert(bundle)
    except Exception as exc:  # skl2onnx/onnxmltools raise several exception types
        logger.info("ONNX conversion failed: %s", exc)
        raise ValueError(
            f"This pipeline could not be converted to ONNX ({exc}). Use the portable bundle format instead."
        ) from exc
    return onnx_model.SerializeToString()


_SCORE_SCRIPT_TEMPLATE = '''"""Standalone scoring script for run {run_key}, exported from Autonomous AI Analyst.

Usage:
    pip install -r requirements.txt
    python score.py input.csv > predictions.csv

Loads the bundled scikit-learn Pipeline with joblib directly — the same
fitted object this application trained, preprocessing included. No network
access and no dependency on the application that produced it beyond what is
pinned in requirements.txt.

joblib.load executes pickle opcodes. Only run this against the
{run_key}_model.pkl file that shipped in this same export — never point it at
a pickle from anywhere else.
"""

import sys

import joblib
import pandas as pd

FEATURES = {features}


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python score.py input.csv", file=sys.stderr)
        raise SystemExit(1)

    bundle = joblib.load("{run_key}_model.pkl")
    pipeline = next(iter(bundle["pipelines"].values()))
    df = pd.read_csv(sys.argv[1])

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {{missing}}")

    predictions = pipeline.predict(df[FEATURES])
    label_encoder = bundle.get("label_encoder")
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions.astype(int))
    pd.DataFrame({{"prediction": predictions}}).to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    main()
'''

_README_TEMPLATE = """# Exported model: {run_key}

Trained by Autonomous AI Analyst. Model: {model_name} ({problem_type}).

## Contents

- `{run_key}_model.pkl` — the fitted scikit-learn Pipeline (preprocessing and
  model together), as a joblib pickle.
- `{run_key}_model.pkl.sig` — an HMAC signature meaningful only inside the
  application that produced it; not needed to use the model here.
- `score.py` — loads the pickle and predicts on a CSV. See its docstring.
- `requirements.txt` — pinned to the versions this server used to fit it.

## A pickle is not a sandbox

`joblib.load` executes pickle opcodes, which is only safe because this file
came from your own run on your own server. Do not replace it with a pickle
from anywhere you would not run arbitrary code from.
"""


def _requirements_for(bundle: dict) -> str:
    import sklearn

    lines = [f"scikit-learn=={sklearn.__version__}", "pandas", "numpy", "joblib"]
    _, _, model = _pipeline_parts(bundle)
    model_name = type(model).__name__ if model is not None else ""
    if "XGB" in model_name:
        import xgboost

        lines.append(f"xgboost=={xgboost.__version__}")
    if "LGBM" in model_name:
        import lightgbm

        lines.append(f"lightgbm=={lightgbm.__version__}")
    return "\n".join(lines) + "\n"


def export_bundle(run_key: str, bundle: dict, metadata: dict, model_path: Path) -> bytes:
    """A zip: the signed pickle, a standalone scoring script, and pinned requirements."""
    model_path = Path(model_path)
    sig_path = Path(str(model_path) + ".sig")
    features = metadata.get("features", [])
    model_name = metadata.get("selected_model", "unknown")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname=f"{run_key}_model.pkl")
        if sig_path.exists():
            zf.write(sig_path, arcname=f"{run_key}_model.pkl.sig")
        zf.writestr("score.py", _SCORE_SCRIPT_TEMPLATE.format(run_key=run_key, features=json.dumps(features)))
        zf.writestr("requirements.txt", _requirements_for(bundle))
        zf.writestr(
            "README.md",
            _README_TEMPLATE.format(
                run_key=run_key, model_name=model_name, problem_type=metadata.get("problem_type", "unknown")
            ),
        )
    return buffer.getvalue()
