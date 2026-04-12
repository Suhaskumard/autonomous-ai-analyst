import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from ml.evaluate import evaluate_classification, evaluate_regression


def detect_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = y.nunique() / max(len(y), 1)
        if y.nunique() <= 20 or unique_ratio < 0.1:
            return "classification"
        return "regression"
    return "classification"


def get_model_registry(problem_type: str) -> dict:
    if problem_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost": XGBClassifier(eval_metric="logloss", n_estimators=50, random_state=42),
            "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "SVM": SVC(probability=True, kernel="rbf", max_iter=2000),
        }
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoostRegressor": XGBRegressor(n_estimators=50, random_state=42),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=10, random_state=42),
        "SVR": SVR(max_iter=2000),
    }


def train_models(X_processed, y, mode: str = "auto", manual_model: str | None = None) -> dict:
    problem_type = detect_problem_type(y)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    registry = get_model_registry(problem_type)

    # In 'auto' mode, skip slow models for large datasets
    if mode == "auto" and len(X_processed) > 5000:
        registry.pop("SVM", None)
        registry.pop("SVR", None)

    if mode == "manual" and manual_model:
        if manual_model not in registry:
            raise ValueError(f"Model '{manual_model}' not available for {problem_type}.")
        registry = {manual_model: registry[manual_model]}

    trained_models = {}
    performance_scores = {}
    failed_models = {}
    for model_name, model in registry.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = (
                evaluate_classification(y_test, y_pred)
                if problem_type == "classification"
                else evaluate_regression(y_test, y_pred)
            )
            trained_models[model_name] = model
            performance_scores[model_name] = metrics
        except Exception as exc:
            failed_models[model_name] = str(exc)

    if not trained_models:
        raise ValueError("All selected models failed to train. Please verify dataset quality and target column.")

    return {
        "problem_type": problem_type,
        "trained_models": trained_models,
        "performance_scores": performance_scores,
        "failed_models": failed_models,
        "X_train": X_train,
    }
