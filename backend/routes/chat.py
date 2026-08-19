"""The analyst endpoint.

What changed in Phase 5, and why:

* **The data no longer leaves the machine.** The snapshot used to be uploaded
  to the Gemini Files API so Google's sandbox could compute on it. Execution is
  now local (`analyst.sandbox`), so the rows stay on this disk and the numbers
  can be checked against the CSV by hand.
* **Tools before code.** Common questions take a deterministic path; only
  genuinely novel ones fall through to generated Python.
* **History is the server's.** The client posts a question and a conversation
  id, not a transcript it could have edited, dropped, or invented.
* **The endpoint is guarded.** A per-conversation message rate and a token
  budget, and a degraded mode that answers honestly when no key is configured.
"""

import json
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

import db
import limits
from analyst import report as report_module
from analyst.agent import run_agent, trim_to_budget
from analyst.providers import Turn, get_provider
from analyst.tools import build_toolset
from auth import Principal, current_principal, owner_scope, require_owned_run
from config import settings
from observability import record_llm_usage
from utils.helpers import METADATA_DIR
from utils.security import artifact_path, validate_dataset_hash
from utils.storage import ensure_local

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_QUERY_CHARS = 4000


class ChatRequest(BaseModel):
    run_key: str | None = None
    # Accepted for backwards compatibility with the pre-Phase-1 field name.
    dataset_hash: str | None = None
    query: str
    conversation_id: str | None = None

    def resolved_key(self) -> str:
        key = self.run_key or self.dataset_hash
        if not key:
            raise HTTPException(status_code=422, detail="run_key is required.")
        return validate_dataset_hash(key)


def _load_metadata(run_key: str) -> dict:
    metadata_path = artifact_path(METADATA_DIR, run_key, ".json")
    # No-op on local storage; fetches from the bucket on S3.
    ensure_local(run_key)
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Run key not found. Upload a dataset first.")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_snapshot(run_key: str):
    """The CSV the analyst computes on, and the frame the tools bind to."""
    import pandas as pd

    snapshot_path = artifact_path(METADATA_DIR, run_key, "_data.csv")
    ensure_local(run_key)
    if not snapshot_path.exists():
        return None, None
    return snapshot_path, pd.read_csv(snapshot_path)


def _system_prompt(metadata: dict, df) -> str:
    health = metadata.get("health", {})
    health_block = ""
    if health:
        reasons = "\n".join(f"- {reason}" for reason in health.get("reasons", []))
        health_block = (
            "### MODEL HEALTH ###\n"
            f"Verdict: {health.get('verdict')} — {health.get('headline')}\n"
            f"{reasons}\n"
            "Never describe this model as reliable if the verdict says otherwise.\n\n"
        )

    columns = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns) if df is not None else "unknown"

    return (
        "You are an autonomous data analyst working on one specific dataset.\n\n"
        "### HOW YOU ANSWER ###\n"
        "You have tools that run on the real rows, on this machine. Use them for every factual\n"
        "claim. Never state a number you did not obtain from a tool result — no estimating, no\n"
        "recalling, no inferring from the schema. If a tool fails, say what failed.\n\n"
        "Prefer the specific tool over run_python: describe_column, correlate (with 'within' to\n"
        "check a relationship inside each segment), filter_rows, group_stats, plot, run_model.\n"
        "Fall back to run_python only when no specific tool fits.\n\n"
        "When you have the results, answer in Markdown: state the finding, give the numbers you\n"
        "obtained, and note any caveat the data implies. Be concise.\n\n"
        "### DATASET ###\n"
        f"Rows: {0 if df is None else len(df):,}\n"
        f"Columns: {columns}\n"
        f"Target: {metadata.get('target')}\n"
        f"Problem type: {metadata.get('problem_type')}\n"
        f"Selected model: {metadata.get('selected_model')}\n\n"
        f"{health_block}"
    )


def _turns_from_history(messages: list[dict]) -> list[Turn]:
    """Stored messages as provider turns. Tool steps are replayed as context."""
    turns: list[Turn] = []
    for message in messages:
        if message["role"] == "user":
            turns.append(Turn(role="user", text=message["content"]))
        elif message["role"] == "assistant":
            summary = message["content"]
            steps = message.get("steps") or []
            if steps:
                rendered = "; ".join(f"{s.get('tool')}: {s.get('summary', '')[:200]}" for s in steps)
                summary = f"{summary}\n\n(Tools used previously — {rendered})"
            turns.append(Turn(role="assistant", text=summary))
    return turns


def _degraded(run_key: str, query: str, conversation_id: str | None, reason: str) -> dict:
    return {
        "run_key": run_key,
        "conversation_id": conversation_id,
        "query": query,
        "response": {
            "answer": reason,
            "steps": [],
            "llm_enabled": False,
            "data_access": "none",
        },
    }


@router.post("/chat")
def chat(req: ChatRequest, principal: Principal = Depends(current_principal)):
    run_key = req.resolved_key()
    require_owned_run(run_key, principal)
    metadata = _load_metadata(run_key)

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="Ask a question.")
    if len(query) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=413, detail=f"Question exceeds {MAX_QUERY_CHARS} characters.")

    # --- conversation ------------------------------------------------------
    conversation_id = req.conversation_id
    if conversation_id:
        conversation = db.get_conversation(conversation_id, owner_id=owner_scope(principal))
        if conversation is None:
            raise HTTPException(status_code=404, detail="Unknown conversation.")
        if conversation["run_key"] != run_key:
            raise HTTPException(status_code=400, detail="That conversation belongs to a different run.")
    else:
        conversation_id = uuid4().hex
        db.create_conversation(conversation_id, run_key, owner_id=principal.user_id)
        conversation = db.get_conversation(conversation_id)

    # --- guards ------------------------------------------------------------
    # Per account first. The per-conversation limits below stop one session
    # running away; only this one stops a client that opens a new conversation
    # for every message, which the per-conversation guard never could.
    limits.check_llm_quota(principal)

    recent = db.count_recent_messages(conversation_id, since_minutes=60)
    if recent >= settings.chat_max_messages_per_hour:
        raise HTTPException(
            status_code=429,
            detail=(
                f"This conversation has used its hourly limit of {settings.chat_max_messages_per_hour} "
                "messages. Start a new conversation or wait."
            ),
        )
    if conversation["total_tokens"] >= settings.chat_token_budget:
        raise HTTPException(
            status_code=429,
            detail=(
                f"This conversation has reached its {settings.chat_token_budget:,}-token budget. "
                "Start a new conversation to continue."
            ),
        )

    provider = get_provider()
    if provider is None:
        db.append_message(conversation_id, "user", query)
        answer = (
            "No language model is configured, so I cannot answer questions. "
            "Set GEMINI_API_KEY in backend/.env and restart. The dashboard, charts, and "
            "predictions all work without it."
        )
        db.append_message(conversation_id, "assistant", answer)
        return _degraded(run_key, query, conversation_id, answer)

    snapshot_path, df = _load_snapshot(run_key)
    if df is None:
        db.append_message(conversation_id, "user", query)
        answer = (
            "The data snapshot for this run is missing from disk, so I have nothing to compute on. "
            "Re-upload the dataset to restore it."
        )
        db.append_message(conversation_id, "assistant", answer)
        return _degraded(run_key, query, conversation_id, answer)

    # --- run ---------------------------------------------------------------
    db.append_message(conversation_id, "user", query)

    history = db.list_messages(conversation_id)
    turns = _turns_from_history(history)
    turns, trimmed = trim_to_budget(turns, settings.chat_token_budget // 2)

    toolset = build_toolset(df, metadata, snapshot_path, run_key, owner_scope(principal) or db.LOCAL_OWNER_ID)
    result = run_agent(provider, _system_prompt(metadata, df), turns, toolset)

    answer = result.answer
    if trimmed:
        answer += "\n\n_Note: earlier turns were dropped from context to stay within the token budget._"

    db.append_message(
        conversation_id,
        "assistant",
        answer,
        steps=[step.as_dict() for step in result.steps],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )

    # Attributable spend: which account, which run, which model. Estimated —
    # see observability.MODEL_RATES for why that word is doing real work.
    estimated_cost = record_llm_usage(
        owner_id=principal.user_id,
        provider=provider.name,
        model=provider.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        run_key=run_key,
        conversation_id=conversation_id,
    )

    return {
        "run_key": run_key,
        "conversation_id": conversation_id,
        "dataset_hash": metadata.get("dataset_hash"),
        "query": query,
        "response": {
            "answer": answer,
            "steps": [step.as_dict() for step in result.steps],
            "llm_enabled": True,
            "llm_provider": provider.name,
            "llm_model": provider.model,
            # Execution is local now, so this is a statement about this machine.
            "data_access": "local-execution",
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "stopped_at_step_limit": result.stopped_at_step_limit,
            "history_trimmed": trimmed,
            "estimated_cost_usd": estimated_cost,
        },
    }


@router.get("/chat/{run_key}/conversations")
def conversations(
    run_key: str,
    limit: int = Query(default=20, ge=1, le=100),
    principal: Principal = Depends(current_principal),
):
    validate_dataset_hash(run_key)
    require_owned_run(run_key, principal)
    return {
        "run_key": run_key,
        "conversations": db.list_conversations(run_key, limit=limit, owner_id=owner_scope(principal)),
    }


@router.get("/chat/conversation/{conversation_id}")
def conversation_messages(conversation_id: str, principal: Principal = Depends(current_principal)):
    conversation = db.get_conversation(conversation_id, owner_id=owner_scope(principal))
    if conversation is None:
        raise HTTPException(status_code=404, detail="Unknown conversation.")
    return {**conversation, "messages": db.list_messages(conversation_id)}


@router.post("/report/{run_key}")
def generate_report(run_key: str, principal: Principal = Depends(current_principal)):
    """One button: a full EDA sweep whose every number is computed, not written."""
    validate_dataset_hash(run_key)
    require_owned_run(run_key, principal)
    # A report is several model calls in one request, so it is the cheapest
    # endpoint to abuse and the one most worth checking the account ceiling on.
    limits.check_llm_quota(principal)
    metadata = _load_metadata(run_key)
    snapshot_path, df = _load_snapshot(run_key)
    if df is None:
        raise HTTPException(status_code=404, detail="No data snapshot for this run. Re-upload the dataset.")

    provider = get_provider()
    report = report_module.build_report(df, metadata, snapshot_path, run_key, provider=provider)
    report["markdown"] = report_module.render_markdown(report)

    # Same accounting as /chat. A report is one large prompt rather than a
    # multi-step agent loop, but it bills to the same key and belongs in the
    # same total — a usage page that only counted chat would understate spend.
    usage = report.pop("usage", None) or {}
    if provider is not None and (usage.get("input_tokens") or usage.get("output_tokens")):
        report["estimated_cost_usd"] = record_llm_usage(
            owner_id=principal.user_id,
            provider=provider.name,
            model=provider.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            run_key=run_key,
        )
    return report
