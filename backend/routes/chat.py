"""The analyst chatbot.

Previously the prompt told Gemini to `pd.read_csv()` a local Windows path.
Code execution runs in Google's remote sandbox with no access to this disk, so
the model answered from summary statistics or invented numbers. The dataset
snapshot is now uploaded to the Gemini Files API and attached to the
conversation, so the sandbox genuinely has the rows it computes on — and when
that upload fails, the model is told plainly that it has statistics only,
rather than being pointed at a path it cannot reach.
"""

import json
import logging
import time

from google import genai
from google.genai import types
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from config import gemini_api_key, gemini_model
from utils.helpers import METADATA_DIR
from utils.security import artifact_path, validate_dataset_hash

logger = logging.getLogger(__name__)

router = APIRouter()

# Gemini keeps uploaded files for 48h; re-uploading the same snapshot on every
# message would be slow and wasteful, so remember the handle for a day.
_FILE_CACHE_TTL_SECONDS = 24 * 60 * 60
_uploaded_files: dict[str, tuple[object, float]] = {}

# Above this, inlining the CSV in the request is not viable and only the Files
# API path can work.
MAX_INLINE_CSV_BYTES = 4 * 1024 * 1024


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    run_key: str | None = None
    # Accepted for backwards compatibility with the pre-Phase-1 field name.
    dataset_hash: str | None = None
    query: str
    history: list[ChatMessage] = Field(default_factory=list)

    def resolved_key(self) -> str:
        key = self.run_key or self.dataset_hash
        if not key:
            raise HTTPException(status_code=422, detail="run_key is required.")
        return validate_dataset_hash(key)


def _get_gemini_client():
    api_key = gemini_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _upload_snapshot(client, run_key: str, snapshot_path):
    """Upload the CSV snapshot to the Files API, cached per run key."""
    cached = _uploaded_files.get(run_key)
    if cached and cached[1] > time.time():
        return cached[0]

    config = {"mime_type": "text/csv", "display_name": f"{run_key[:12]}_snapshot.csv"}
    try:
        uploaded = client.files.upload(file=str(snapshot_path), config=config)
    except TypeError:
        # Older SDK signature.
        uploaded = client.files.upload(path=str(snapshot_path), config=config)

    # A freshly uploaded file is briefly PROCESSING and cannot be referenced yet.
    deadline = time.time() + 30
    while getattr(getattr(uploaded, "state", None), "name", "ACTIVE") == "PROCESSING" and time.time() < deadline:
        time.sleep(0.5)
        uploaded = client.files.get(name=uploaded.name)

    state = getattr(getattr(uploaded, "state", None), "name", "ACTIVE")
    if state == "FAILED":
        raise RuntimeError("Gemini reported the uploaded snapshot as FAILED.")

    _uploaded_files[run_key] = (uploaded, time.time() + _FILE_CACHE_TTL_SECONDS)
    return uploaded


def _build_system_prompt(metadata: dict, data_access: str, filename: str | None) -> str:
    summary_stats = metadata.get("summary_stats", {})
    features = metadata.get("features", [])
    health = metadata.get("health", {})

    if data_access == "file":
        access_block = (
            "### DATA ACCESS ###\n"
            f"The full dataset is ATTACHED to this conversation as a CSV file ('{filename}').\n"
            "Load it in your Python environment to compute real answers. Do NOT invent a local\n"
            "file path, and do NOT answer numerically from the summary statistics when the\n"
            "attached data can be computed on directly.\n"
        )
    else:
        access_block = (
            "### DATA ACCESS ###\n"
            "You do NOT have the raw rows in this conversation — only the summary statistics\n"
            "below. If a question needs the underlying rows, say so explicitly instead of\n"
            "estimating, and never claim to have executed code against the dataset.\n"
        )

    health_block = ""
    if health:
        health_block = (
            "### MODEL HEALTH ###\n"
            f"Verdict: {health.get('verdict')} — {health.get('headline')}\n"
            f"{chr(10).join('- ' + r for r in health.get('reasons', []))}\n"
            "Do not describe this model as reliable if the verdict says otherwise.\n\n"
        )

    return (
        "You are an Advanced Autonomous AI Data Analyst with Python code execution.\n\n"
        f"{access_block}\n"
        "### DATASET CONTEXT ###\n"
        f"Input Features: {', '.join(map(str, features))}\n"
        f"Target Column: {metadata.get('target')}\n"
        f"Problem Type: {metadata.get('problem_type')}\n"
        f"Selected Model: {metadata.get('selected_model')}\n\n"
        f"{health_block}"
        "### SUMMARY STATISTICS ###\n"
        f"{json.dumps(summary_stats, indent=2)[:20000]}\n\n"
        "### INSTRUCTIONS ###\n"
        "1. Use code execution for anything requiring a precise number.\n"
        "2. Show the code you ran and the result it produced, so the user can verify it.\n"
        "3. State your assumptions; if you cannot compute something, say why.\n"
        "4. Never present an estimate as a computed result.\n"
        "5. Use Markdown formatting.\n"
    )


def _extract_answer(response) -> str:
    """Walk the response parts.

    With code execution enabled a reply can be entirely executable-code and
    result parts with no plain text at all, in which case `response.text` is
    None and `.strip()` used to raise straight into the user's face.
    """
    try:
        direct = response.text
        if direct and direct.strip():
            return direct.strip()
    except Exception:  # noqa: BLE001 - .text raises on some part combinations
        logger.info("response.text was unavailable; walking response parts instead.")

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text.strip())
                continue
            code = getattr(part, "executable_code", None)
            if code is not None and getattr(code, "code", None):
                chunks.append(f"```python\n{code.code.strip()}\n```")
                continue
            outcome = getattr(part, "code_execution_result", None)
            if outcome is not None and getattr(outcome, "output", None):
                chunks.append(f"**Result**\n```\n{outcome.output.strip()}\n```")

    if chunks:
        return "\n\n".join(chunks)

    finish = None
    for candidate in getattr(response, "candidates", None) or []:
        finish = getattr(candidate, "finish_reason", None) or finish
    if finish:
        return f"The model returned no readable content (finish reason: {finish})."
    return "The model returned an empty response. Try rephrasing the question."


@router.post("/chat")
def chat(req: ChatRequest):
    run_key = req.resolved_key()
    metadata_path = artifact_path(METADATA_DIR, run_key, ".json")
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Run key not found. Upload a dataset first.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    client = _get_gemini_client()
    if not client:
        return {
            "run_key": run_key,
            "query": req.query,
            "response": {
                "answer": "Chatbot is unavailable because GEMINI_API_KEY is not set.",
                "llm_enabled": False,
                "data_access": "none",
            },
        }

    model_id = gemini_model()
    snapshot_path = artifact_path(METADATA_DIR, run_key, "_data.csv")

    # Attach the real rows. Files API first; a small snapshot can also ride
    # along inline if the upload fails.
    attachment = None
    data_access = "stats-only"
    attachment_name = None
    if snapshot_path.exists():
        try:
            uploaded = _upload_snapshot(client, run_key, snapshot_path)
            attachment = uploaded
            attachment_name = getattr(uploaded, "display_name", None) or getattr(uploaded, "name", "dataset.csv")
            data_access = "file"
        except Exception as exc:
            logger.warning("Gemini Files API upload failed for %s: %s", run_key, exc, exc_info=True)
            _uploaded_files.pop(run_key, None)
            size = snapshot_path.stat().st_size
            if size <= MAX_INLINE_CSV_BYTES:
                try:
                    attachment = types.Part.from_bytes(
                        data=snapshot_path.read_bytes(), mime_type="text/csv"
                    )
                    attachment_name = "dataset.csv"
                    data_access = "file"
                except Exception as inline_exc:
                    logger.warning("Inline CSV attachment also failed: %s", inline_exc)
    else:
        logger.warning("No data snapshot on disk for run %s; answering from statistics only.", run_key)

    system_prompt = _build_system_prompt(metadata, data_access, attachment_name)

    history = [
        types.Content(
            role="user" if msg.role == "user" else "model",
            parts=[types.Part(text=msg.content)],
        )
        for msg in req.history
    ]

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        )
        chat_session = client.chats.create(model=model_id, history=history, config=config)
        message = [attachment, req.query] if attachment is not None else req.query
        response = chat_session.send_message(message)
        answer = _extract_answer(response)
    except Exception as exc:
        logger.exception("Gemini request failed for run %s", run_key)
        answer = f"Error generating response: {exc}"
        data_access = data_access if attachment is not None else "stats-only"

    return {
        "run_key": run_key,
        "dataset_hash": metadata.get("dataset_hash"),
        "query": req.query,
        "response": {
            "answer": answer,
            "llm_enabled": True,
            "llm_provider": "gemini",
            "llm_model": model_id,
            # The UI shows this, so the user always knows whether the answer
            # was computed on real rows or inferred from statistics.
            "data_access": data_access,
        },
    }
