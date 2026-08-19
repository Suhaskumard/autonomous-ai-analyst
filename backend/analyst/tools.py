"""The analyst's tools.

Free-form code generation for every question is expensive, slow, and only
probably correct. Most of what people actually ask a data analyst — describe
this column, what correlates with what, show me the rows where X, plot Y — is
a fixed computation with a parameter or two. Those get a deterministic
implementation here: no code generation, no sandbox spawn, the same answer
every time.

`run_python` is the escape hatch underneath. Genuinely novel questions still
fall through to generated code in the sandbox; the tools just mean that is the
exception rather than the path for "what is the mean of age".

Every tool returns `{ok, summary, ...payload}`. `summary` is the compact text
the model reads back; the rest is structure the UI renders.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from analyst import sandbox

logger = logging.getLogger(__name__)

MAX_ROWS_RETURNED = 100
MAX_CORRELATION_COLUMNS = 25
MAX_SEGMENTS = 12


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[..., dict]


def _table(frame: pd.DataFrame, limit: int = MAX_ROWS_RETURNED) -> dict:
    view = frame.head(limit)
    return {
        "kind": "table",
        "columns": [str(c) for c in view.columns],
        "rows": [[_clean(v) for v in row] for row in view.itertuples(index=False, name=None)],
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "truncated": bool(len(frame) > limit),
    }


def _clean(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return str(value)


def _fail(message: str) -> dict:
    return {"ok": False, "summary": message, "error": message}


def _require_column(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns:
        options = ", ".join(map(str, df.columns[:25]))
        return f"There is no column called '{column}'. Available columns: {options}."
    return None


# --- the tools --------------------------------------------------------------


def describe_column(df: pd.DataFrame, column: str) -> dict:
    """Type, missingness, and the right summary for that type."""
    problem = _require_column(df, column)
    if problem:
        return _fail(problem)

    series = df[column]
    missing = int(series.isna().sum())
    payload = {
        "ok": True,
        "column": column,
        "missing": missing,
        "missing_pct": round(missing / max(len(series), 1) * 100, 2),
        "distinct": int(series.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(series):
        described = series.describe()
        stats = {key: _clean(described[key]) for key in ("mean", "std", "min", "25%", "50%", "75%", "max")}
        payload.update({"dtype": "numeric", "stats": stats})
        payload["summary"] = (
            f"{column} is numeric: mean {stats['mean']}, median {stats['50%']}, "
            f"range {stats['min']} to {stats['max']}, sd {stats['std']}, "
            f"{missing} missing ({payload['missing_pct']}%), {payload['distinct']} distinct values."
        )
    else:
        counts = series.astype(str).value_counts().head(10)
        payload.update(
            {
                "dtype": "categorical",
                "top_values": [{"value": str(k), "count": int(v)} for k, v in counts.items()],
            }
        )
        top = ", ".join(f"{k} ({v})" for k, v in counts.head(5).items())
        payload["summary"] = (
            f"{column} is categorical with {payload['distinct']} distinct values. "
            f"Most common: {top}. {missing} missing ({payload['missing_pct']}%)."
        )
    return payload


def correlate(df: pd.DataFrame, columns: list[str] | None = None, within: str | None = None, top: int = 5) -> dict:
    """Pearson correlations, optionally computed separately within each segment.

    `within` is what makes "does that hold within each segment?" a single
    deterministic call rather than a code-generation problem.
    """
    numeric = df.select_dtypes(include="number")
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return _fail(f"Unknown column(s): {', '.join(missing)}.")
        numeric = df[columns].select_dtypes(include="number")

    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return _fail("Correlation needs at least two numeric columns that actually vary.")
    if numeric.shape[1] > MAX_CORRELATION_COLUMNS:
        numeric = numeric.iloc[:, :MAX_CORRELATION_COLUMNS]

    overall = _pair_list(numeric, top)
    payload: dict = {
        "ok": True,
        "columns": [str(c) for c in numeric.columns],
        "matrix": _table(numeric.corr(numeric_only=True).round(4).reset_index().rename(columns={"index": "column"})),
        "top_pairs": overall,
    }

    lines = ["Strongest correlations overall: " + "; ".join(f"{p['a']} vs {p['b']} r={p['r']}" for p in overall[:top])]

    if within:
        problem = _require_column(df, within)
        if problem:
            return _fail(problem)
        segments = df[within].astype(str).value_counts().head(MAX_SEGMENTS).index.tolist()
        headline = overall[0] if overall else None
        by_segment = []
        for segment in segments:
            subset = df[df[within].astype(str) == segment]
            if len(subset) < 3:
                by_segment.append({"segment": segment, "n": int(len(subset)), "r": None, "note": "too few rows"})
                continue
            pair_r = None
            if headline:
                a, b = headline["a"], headline["b"]
                if a in subset and b in subset and subset[a].nunique() > 1 and subset[b].nunique() > 1:
                    pair_r = round(float(subset[a].corr(subset[b])), 4)
            by_segment.append({"segment": segment, "n": int(len(subset)), "r": pair_r})

        payload["within"] = within
        payload["headline_pair"] = headline
        payload["by_segment"] = by_segment
        if headline:
            rendered = "; ".join(
                f"{row['segment']} (n={row['n']}): " + ("n/a" if row["r"] is None else str(row["r"]))
                for row in by_segment
            )
            lines.append(
                f"The strongest pair ({headline['a']} vs {headline['b']}, r={headline['r']} overall) "
                f"within each '{within}': {rendered}"
            )

    payload["summary"] = "\n".join(lines)
    return payload


def _pair_list(numeric: pd.DataFrame, top: int) -> list[dict]:
    matrix = numeric.corr(numeric_only=True)
    pairs = []
    columns = list(matrix.columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            value = matrix.loc[a, b]
            if pd.notna(value):
                pairs.append({"a": str(a), "b": str(b), "r": round(float(value), 4)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return pairs[: max(top, 1)]


def filter_rows(df: pd.DataFrame, expression: str, limit: int = 20, columns: list[str] | None = None) -> dict:
    """Rows matching a pandas query expression.

    `DataFrame.query` rather than `eval` of arbitrary Python: it parses a
    restricted comparison grammar, so a filter cannot become code execution.
    """
    try:
        matched = df.query(expression)
    except Exception as exc:
        return _fail(f"Could not apply the filter `{expression}`: {exc}")

    if columns:
        keep = [c for c in columns if c in matched.columns]
        if keep:
            matched = matched[keep]

    limit = max(1, min(int(limit), MAX_ROWS_RETURNED))
    return {
        "ok": True,
        "expression": expression,
        "matched": int(len(matched)),
        "table": _table(matched, limit),
        "summary": (
            f"{len(matched):,} of {len(df):,} rows match `{expression}` "
            f"({len(matched) / max(len(df), 1):.1%}). Showing up to {limit}."
        ),
    }


def group_stats(df: pd.DataFrame, by: str, column: str, agg: str = "mean") -> dict:
    """One aggregate of `column` per level of `by` — the segment comparison."""
    for candidate in (by, column):
        problem = _require_column(df, candidate)
        if problem:
            return _fail(problem)

    allowed = {"mean", "median", "sum", "min", "max", "count", "std", "nunique"}
    if agg not in allowed:
        return _fail(f"Unsupported aggregate '{agg}'. Use one of: {', '.join(sorted(allowed))}.")
    if agg not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(df[column]):
        return _fail(f"'{column}' is not numeric, so '{agg}' does not apply. Try count or nunique.")

    grouped = df.groupby(df[by].astype(str))[column].agg(agg).sort_values(ascending=False)
    grouped = grouped.head(MAX_SEGMENTS * 2)
    frame = grouped.reset_index()
    frame.columns = [by, f"{agg}_{column}"]

    rendered = "; ".join(f"{row[0]}: {_clean(row[1])}" for row in frame.head(10).itertuples(index=False, name=None))
    return {
        "ok": True,
        "by": by,
        "column": column,
        "agg": agg,
        "table": _table(frame),
        "summary": f"{agg} of {column} by {by} — {rendered}",
    }


def plot(df: pd.DataFrame, kind: str, x: str, y: str | None = None, snapshot_path=None, bins: int = 20) -> dict:
    """Draw a chart in the sandbox, so plotting shares one execution path."""
    allowed = {"hist", "scatter", "bar", "box", "line"}
    if kind not in allowed:
        return _fail(f"Unsupported plot kind '{kind}'. Use one of: {', '.join(sorted(allowed))}.")
    problem = _require_column(df, x)
    if problem:
        return _fail(problem)
    if y is not None:
        problem = _require_column(df, y)
        if problem:
            return _fail(problem)
    if kind in {"scatter", "line"} and y is None:
        return _fail(f"A {kind} plot needs both x and y.")

    code = _plot_code(kind, x, y, bins)
    result = sandbox.run_code(code, snapshot_path)
    if not result.ok:
        return _fail(f"The plot could not be drawn: {result.error}")
    return {
        "ok": True,
        "kind": kind,
        "figures": result.figures,
        "code": code,
        "summary": f"Drew a {kind} plot of {x}" + (f" against {y}." if y else ".") + " The image is shown to the user.",
    }


def _plot_code(kind: str, x: str, y: str | None, bins: int) -> str:
    xr, yr = repr(x), repr(y)
    header = "fig, ax = plt.subplots(figsize=(7, 4))\n"
    body = {
        "hist": f"df[{xr}].plot(kind='hist', bins={int(bins)}, ax=ax)",
        "box": f"df[[{xr}]].plot(kind='box', ax=ax)",
        "bar": f"df[{xr}].astype(str).value_counts().head(20).plot(kind='bar', ax=ax)",
        "scatter": f"df.plot(kind='scatter', x={xr}, y={yr}, alpha=0.6, ax=ax)",
        "line": f"df.sort_values({xr}).plot(kind='line', x={xr}, y={yr}, ax=ax)",
    }[kind]
    title = f"ax.set_title({repr(f'{kind}: {x}' + (f' vs {y}' if y else ''))})\n"
    return header + body + "\n" + title + "fig.tight_layout()"


def run_model(metadata: dict, rows: list[dict], run_key: str, owner_id: str | None = None) -> dict:
    """Predict with the run's own fitted model, on rows the user supplies.

    `owner_id` is threaded through because this goes down the same path a direct
    prediction does, logging included. Without it the agent's predictions would
    be recorded against the implicit local owner while the caller was somebody
    else — invisible in that account's monitoring, and mixed into another's.
    """
    if not rows:
        return _fail("Provide at least one row to predict on.")

    from routes.predict import predict_rows

    try:
        prediction = predict_rows(run_key, pd.DataFrame(rows), owner_id=owner_id)
    except Exception as exc:
        return _fail(f"Prediction failed: {exc}")

    return {
        "ok": True,
        "table": _table(pd.DataFrame(prediction["rows"])),
        "summary": (
            f"Predicted {metadata.get('target')} for {len(prediction['rows'])} row(s) using the stored "
            f"{metadata.get('selected_model')} model: "
            + ", ".join(str(r.get("prediction")) for r in prediction["rows"][:10])
        ),
    }


def run_python(code: str, snapshot_path) -> dict:
    """The escape hatch: arbitrary pandas in the sandbox, for novel questions."""
    result = sandbox.run_code(code, snapshot_path)
    return {
        "ok": result.ok,
        "execution": result.as_dict(),
        "figures": result.figures,
        "summary": result.summarise(),
    }


# --- schemas the provider advertises ---------------------------------------


def build_toolset(
    df: pd.DataFrame, metadata: dict, snapshot_path, run_key: str, owner_id: str | None = None
) -> dict[str, Tool]:
    """Bind the tools to one run's data. Names/schemas are provider-agnostic."""
    columns = [str(c) for c in df.columns]
    column_enum = {"type": "string", "description": f"One of: {', '.join(columns[:40])}"}

    return {
        "describe_column": Tool(
            name="describe_column",
            description=(
                "Summary statistics for one column: type, missing count, distinct count, and either "
                "mean/median/quartiles (numeric) or the most common values (categorical)."
            ),
            parameters={
                "type": "object",
                "properties": {"column": column_enum},
                "required": ["column"],
            },
            run=lambda column: describe_column(df, column),
        ),
        "correlate": Tool(
            name="correlate",
            description=(
                "Pearson correlations between numeric columns, with the strongest pairs ranked. "
                "Pass 'within' to also recompute the strongest pair separately inside each level of "
                "that column — use this to check whether a relationship holds per segment."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to these columns. Omit to use every numeric column.",
                    },
                    "within": {
                        "type": "string",
                        "description": "Group column; the top pair is recomputed inside each of its levels.",
                    },
                    "top": {"type": "integer", "description": "How many pairs to rank (default 5)."},
                },
            },
            run=lambda columns=None, within=None, top=5: correlate(df, columns, within, top),
        ),
        "filter_rows": Tool(
            name="filter_rows",
            description=(
                "Rows matching a pandas query expression, e.g. \"age > 50 and region == 'north'\". "
                "Returns the match count and a sample of the rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A pandas .query() expression."},
                    "limit": {"type": "integer", "description": "Rows to return (default 20, max 100)."},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to keep."},
                },
                "required": ["expression"],
            },
            run=lambda expression, limit=20, columns=None: filter_rows(df, expression, limit, columns),
        ),
        "group_stats": Tool(
            name="group_stats",
            description="One aggregate of a numeric column per level of a grouping column.",
            parameters={
                "type": "object",
                "properties": {
                    "by": column_enum,
                    "column": column_enum,
                    "agg": {
                        "type": "string",
                        "description": "mean, median, sum, min, max, count, std or nunique (default mean).",
                    },
                },
                "required": ["by", "column"],
            },
            run=lambda by, column, agg="mean": group_stats(df, by, column, agg),
        ),
        "plot": Tool(
            name="plot",
            description="Draw a chart (hist, box, bar, scatter, line) and show it to the user.",
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "description": "hist, box, bar, scatter or line."},
                    "x": column_enum,
                    "y": {"type": "string", "description": "Second column, required for scatter and line."},
                    "bins": {"type": "integer", "description": "Histogram bins (default 20)."},
                },
                "required": ["kind", "x"],
            },
            run=lambda kind, x, y=None, bins=20: plot(df, kind, x, y, snapshot_path, bins),
        ),
        "run_model": Tool(
            name="run_model",
            description=(
                f"Predict {metadata.get('target')} for supplied feature rows using this run's fitted "
                f"{metadata.get('selected_model')} model."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Rows as objects keyed by feature name.",
                    }
                },
                "required": ["rows"],
            },
            run=lambda rows: run_model(metadata, rows, run_key, owner_id),
        ),
        "run_python": Tool(
            name="run_python",
            description=(
                "Execute Python against the dataset when no other tool fits. `df` is the DataFrame; "
                "pandas is `pd`, numpy is `np`, matplotlib is `plt`. The value of the last expression "
                "is captured, as is anything printed. No network, no file writes, no subprocesses."
            ),
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python to execute."}},
                "required": ["code"],
            },
            run=lambda code: run_python(code, snapshot_path),
        ),
    }


def execute_tool(toolset: dict[str, Tool], name: str, arguments: dict) -> dict:
    """Dispatch one call, turning every failure into a result the model can read."""
    tool = toolset.get(name)
    if tool is None:
        return _fail(f"There is no tool called '{name}'. Available: {', '.join(sorted(toolset))}.")
    try:
        return tool.run(**(arguments or {}))
    except TypeError as exc:
        return _fail(f"Wrong arguments for {name}: {exc}")
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return _fail(f"{name} failed: {exc}")
