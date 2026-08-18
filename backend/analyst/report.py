"""Autonomous EDA report.

One button, one sweep, one shareable document. The important design choice is
that **every number in it is computed, not generated**: the sweep runs the same
deterministic tools the analyst uses, and the LLM is asked only to write the
narrative around results it is handed. If there is no LLM configured, the
report still builds — it simply has no prose, which is a better failure than
prose with no numbers.
"""

import logging
from datetime import UTC, datetime

import pandas as pd

from analyst import tools as tool_module
from analyst.providers import LLMProvider, Turn

logger = logging.getLogger(__name__)

MAX_PROFILED_COLUMNS = 12
MAX_PLOTS = 4

NARRATIVE_SYSTEM = (
    "You are a data analyst writing the summary section of an exploratory report.\n"
    "You are given the computed findings as JSON. Write 3-6 short paragraphs in Markdown:\n"
    "what the dataset contains, the relationships that stand out, anything that looks like a\n"
    "data-quality problem, and what the model result means.\n\n"
    "Rules:\n"
    "1. Every number you state must come from the findings given to you. Do not compute or estimate.\n"
    "2. If the model health verdict is weak, say so plainly rather than describing the model as good.\n"
    "3. Note limitations you can see in the findings (small classes, high missingness, weak correlations).\n"
    "4. No preamble, no headings above level 3, no bullet-point-only answers.\n"
)


def build_report(
    df: pd.DataFrame,
    metadata: dict,
    snapshot_path,
    run_key: str,
    provider: LLMProvider | None = None,
) -> dict:
    """Run the sweep, then optionally narrate it."""
    toolset = tool_module.build_toolset(df, metadata, snapshot_path, run_key)
    sections: list[dict] = []

    sections.append(_overview_section(df, metadata))
    sections.append(_quality_section(df, metadata))

    profiles = _profiles_section(df, toolset)
    if profiles:
        sections.append(profiles)

    correlations = _correlations_section(df, metadata, toolset)
    if correlations:
        sections.append(correlations)

    segments = _segments_section(df, metadata, toolset)
    if segments:
        sections.append(segments)

    charts = _charts_section(df, toolset)
    if charts:
        sections.append(charts)

    sections.append(_model_section(metadata))

    findings = {section["title"]: section.get("facts", section.get("summary")) for section in sections}
    narrative, narrative_error = _narrate(provider, metadata, findings)

    return {
        "run_key": run_key,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target": metadata.get("target"),
            "problem_type": metadata.get("problem_type"),
        },
        "narrative": narrative,
        "narrative_error": narrative_error,
        "sections": sections,
    }


# --- sections ---------------------------------------------------------------


def _overview_section(df: pd.DataFrame, metadata: dict) -> dict:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return {
        "id": "overview",
        "title": "Overview",
        "summary": (
            f"{df.shape[0]:,} rows and {df.shape[1]} columns "
            f"({len(numeric)} numeric, {len(categorical)} categorical). "
            f"Target: {metadata.get('target')} ({metadata.get('problem_type')})."
        ),
        "facts": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_columns": numeric[:40],
            "categorical_columns": categorical[:40],
            "target": metadata.get("target"),
            "problem_type": metadata.get("problem_type"),
        },
    }


def _quality_section(df: pd.DataFrame, metadata: dict) -> dict:
    missing = df.isna().sum()
    worst = missing[missing > 0].sort_values(ascending=False).head(10)
    duplicates = int(df.duplicated().sum())
    constant = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]

    notes = []
    if len(worst):
        notes.append(
            "Columns with missing values: "
            + ", ".join(f"{col} ({int(n)}, {n / len(df):.1%})" for col, n in worst.items())
        )
    if duplicates:
        notes.append(f"{duplicates:,} duplicate rows.")
    if constant:
        notes.append(f"Constant columns carrying no signal: {', '.join(constant)}.")
    notes.extend(metadata.get("quality_report", {}).get("warnings", []))

    return {
        "id": "quality",
        "title": "Data quality",
        "summary": " ".join(notes) if notes else "No missing values, duplicates, or constant columns.",
        "facts": {
            "missing_by_column": {str(k): int(v) for k, v in worst.items()},
            "duplicate_rows": duplicates,
            "constant_columns": constant,
        },
    }


def _profiles_section(df: pd.DataFrame, toolset: dict) -> dict | None:
    profiles = []
    for column in list(df.columns)[:MAX_PROFILED_COLUMNS]:
        result = tool_module.execute_tool(toolset, "describe_column", {"column": str(column)})
        if result.get("ok"):
            profiles.append({"column": str(column), "summary": result["summary"], "dtype": result.get("dtype")})
    if not profiles:
        return None
    return {
        "id": "columns",
        "title": "Column profiles",
        "summary": f"Profiled {len(profiles)} column(s).",
        "facts": {entry["column"]: entry["summary"] for entry in profiles},
        "profiles": profiles,
    }


def _correlations_section(df: pd.DataFrame, metadata: dict, toolset: dict) -> dict | None:
    result = tool_module.execute_tool(toolset, "correlate", {"top": 5})
    if not result.get("ok"):
        return None
    return {
        "id": "correlations",
        "title": "Relationships",
        "summary": result["summary"],
        "facts": {"top_pairs": result.get("top_pairs", [])},
        "table": result.get("matrix"),
    }


def _segments_section(df: pd.DataFrame, metadata: dict, toolset: dict) -> dict | None:
    """Does the headline relationship hold inside each segment?

    The report answers the exit-criterion question unprompted, because it is
    the question a real analyst asks next and almost nobody asks first.
    """
    categorical = [
        str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and 1 < df[c].nunique(dropna=True) <= 12
    ]
    if not categorical:
        return None

    segment_column = categorical[0]
    result = tool_module.execute_tool(toolset, "correlate", {"within": segment_column, "top": 3})
    if not result.get("ok") or not result.get("by_segment"):
        return None

    return {
        "id": "segments",
        "title": f"Consistency within {segment_column}",
        "summary": result["summary"],
        "facts": {
            "segment_column": segment_column,
            "headline_pair": result.get("headline_pair"),
            "by_segment": result.get("by_segment", []),
        },
    }


def _charts_section(df: pd.DataFrame, toolset: dict) -> dict | None:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]

    requests: list[dict] = [{"kind": "hist", "x": str(col)} for col in numeric[:2]]
    if len(numeric) >= 2:
        requests.append({"kind": "scatter", "x": str(numeric[0]), "y": str(numeric[1])})
    if categorical:
        requests.append({"kind": "bar", "x": str(categorical[0])})

    figures = []
    for request in requests[:MAX_PLOTS]:
        result = tool_module.execute_tool(toolset, "plot", request)
        if result.get("ok"):
            for figure in result.get("figures", []):
                figures.append({**figure, "caption": result["summary"]})

    if not figures:
        return None
    return {
        "id": "charts",
        "title": "Distributions",
        "summary": f"{len(figures)} chart(s) drawn from the snapshot.",
        "figures": figures,
    }


def _model_section(metadata: dict) -> dict:
    health = metadata.get("health", {})
    selected = metadata.get("selected_model")
    cv = (metadata.get("cv_scores", {}).get(selected, {}) or {}).get(metadata.get("selection_metric", ""), {})
    holdout = metadata.get("all_model_scores", {}).get(selected, {})

    summary = f"{selected} was selected on cross-validated {metadata.get('selection_metric')}."
    if cv:
        summary += f" CV {cv.get('mean'):.4g} ± {cv.get('std'):.4g}."
    if health:
        summary += f" Verdict: {health.get('verdict')} — {health.get('headline')}."

    return {
        "id": "model",
        "title": "Model result",
        "summary": summary,
        "facts": {
            "selected_model": selected,
            "selection_metric": metadata.get("selection_metric"),
            "cross_validated": cv,
            "holdout": holdout,
            "health": health,
            "per_class": metadata.get("per_class", []),
            "imbalance_warnings": metadata.get("imbalance", {}).get("warnings", []),
        },
    }


# --- narrative --------------------------------------------------------------


def _narrate(provider: LLMProvider | None, metadata: dict, findings: dict) -> tuple[str, str | None]:
    if provider is None:
        return (
            "",
            "No language model is configured, so this report has computed findings but no written summary.",
        )
    import json

    prompt = (
        f"Dataset target: {metadata.get('target')} ({metadata.get('problem_type')}).\n\n"
        f"Computed findings:\n{json.dumps(findings, default=str)[:20000]}"
    )
    try:
        reply = provider.complete(NARRATIVE_SYSTEM, [Turn(role="user", text=prompt)], [])
        return reply.text.strip(), None
    except Exception as exc:
        logger.warning("Report narrative failed: %s", exc)
        return "", f"The written summary could not be generated ({exc}). The computed findings below are unaffected."


def render_markdown(report: dict) -> str:
    """The shareable artefact: one self-contained Markdown document."""
    dataset = report.get("dataset", {})
    lines = [
        f"# Analysis report — {dataset.get('target') or 'dataset'}",
        "",
        f"*Generated {report.get('generated_at')} · {dataset.get('rows'):,} rows × {dataset.get('columns')} columns "
        f"· {dataset.get('problem_type')}*",
        "",
    ]

    if report.get("narrative"):
        lines += ["## Summary", "", report["narrative"], ""]
    elif report.get("narrative_error"):
        lines += ["## Summary", "", f"> {report['narrative_error']}", ""]

    for section in report.get("sections", []):
        lines += [f"## {section['title']}", ""]
        if section.get("summary"):
            lines += [section["summary"], ""]
        for profile in section.get("profiles", []) or []:
            lines.append(f"- **{profile['column']}** — {profile['summary']}")
        if section.get("profiles"):
            lines.append("")
        table = section.get("table")
        if table:
            lines += _markdown_table(table) + [""]
        if section.get("figures"):
            lines += [f"*({len(section['figures'])} chart(s) — see the interactive report)*", ""]

    return "\n".join(lines)


def _markdown_table(table: dict) -> list[str]:
    columns = table.get("columns", [])
    if not columns:
        return []
    rows = table.get("rows", [])[:25]
    out = ["| " + " | ".join(str(c) for c in columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows]
    return out
