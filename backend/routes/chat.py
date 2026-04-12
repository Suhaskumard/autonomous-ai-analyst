import json
import os

from google import genai
from google.genai import types
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.helpers import METADATA_DIR

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    dataset_hash: str
    query: str
    history: list[ChatMessage] = []


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


@router.post("/chat")
def chat(req: ChatRequest):
    metadata_path = METADATA_DIR / f"{req.dataset_hash}.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Dataset hash not found. Upload dataset first.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    client = _get_gemini_client()
    if not client:
        return {
            "dataset_hash": req.dataset_hash,
            "query": req.query,
            "response": {
                "answer": "Chatbot is unavailable because GEMINI_API_KEY is not set.",
                "llm_enabled": False,
            },
        }

    model_id = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # Prepare context
    summary_stats = metadata.get("summary_stats", {})
    features = metadata.get("features", [])
    target = metadata.get("target")
    data_path = metadata.get("data_snapshot_path")
    
    # Format history for Gemini
    history = []
    for msg in req.history:
        history.append(types.Content(
            role="user" if msg.role == "user" else "model",
            parts=[types.Part(text=msg.content)]
        ))

    system_prompt = (
        "You are an Advanced Autonomous AI Data Analyst.\n"
        "You have been granted CODE EXECUTION capabilities to perform precise data analysis.\n\n"
        "### DATASET CONTEXT ###\n"
        f"File Path: {data_path}\n"
        f"Input Features: {', '.join(features)}\n"
        f"Target Column: {target}\n"
        f"Problem Type: {metadata.get('problem_type')}\n\n"
        "### SUMMARY STATISTICS ###\n"
        f"{json.dumps(summary_stats, indent=2)}\n\n"
        "### INSTRUCTIONS ###\n"
        "1. You MUST use Python code execution to answer complex questions or when precise calculations are needed.\n"
        "2. The dataset is located at the 'File Path' provided above. Use `pd.read_csv()` to load it.\n"
        "3. You can generate charts by writing code; describe what the chart would look like or provide the underlying data.\n"
        "4. Be proactive. If a user asks for 'insights', run analyses on correlations, outliers, or feature importance using the actual data.\n"
        "5. ALWAYS provide a clear, professional summary of your findings after running code.\n"
        "6. Use Markdown formatting for your responses.\n"
    )

    try:
        # The new SDK supports system_instruction directly in the config
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )
        chat_session = client.chats.create(model=model_id, history=history, config=config)
        response = chat_session.send_message(req.query)
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    return {
        "dataset_hash": req.dataset_hash,
        "query": req.query,
        "response": {
            "answer": answer,
            "llm_enabled": True,
            "llm_provider": "gemini",
            "llm_model": model_id,
        },
    }
