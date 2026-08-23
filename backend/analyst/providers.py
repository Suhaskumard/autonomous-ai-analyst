"""The LLM behind an interface.

`chat.py` used to construct a Gemini client inline, so the provider was
untestable without a network call and unswappable without a rewrite. Every
provider here answers the same question — given a system prompt, a
conversation, and a set of tools, what is the next reply — and returns the
same `LLMReply` whether it came from Gemini or from a scripted fake.

The fake is what lets the agent loop, the tool dispatch, the token budget and
the whole chat route be tested deterministically and offline.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    # Providers that correlate calls to results by id supply one.
    call_id: str | None = None
    # Gemini's "thinking" models (e.g. gemini-3-flash-preview) attach an opaque
    # signature to each function call and reject the next turn if a replayed
    # call is missing it. Only Gemini sets this; other providers leave it None.
    thought_signature: bytes | None = None


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Turn:
    """One entry in the conversation as the provider should see it."""

    role: str  # "user" | "assistant" | "tool"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None
    tool_result: dict | None = None


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, turns: list[Turn], tools: list[Any]) -> LLMReply: ...


class ProviderUnavailable(RuntimeError):
    """No usable provider — no key, or the SDK is missing."""


# --- Gemini -----------------------------------------------------------------


class GeminiProvider:
    """google-genai with function calling.

    Tools are declared rather than left to free-form code execution: the model
    picks a named tool and this process runs it locally, against the snapshot
    on this disk. Google's own sandbox is no longer in the loop, which is what
    makes the arithmetic verifiable here.
    """

    name = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def complete(self, system: str, turns: list[Turn], tools: list[Any]) -> LLMReply:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=_to_gemini_schema(tool.parameters),
            )
            for tool in tools
        ]

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)] if declarations else None,
            # Deterministic-ish: an analyst restating a number should restate
            # the same number.
            temperature=0.2,
        )

        response = self._client.models.generate_content(
            model=self.model,
            contents=[_to_gemini_content(turn) for turn in turns],
            config=config,
        )
        return _from_gemini_response(response)


def _to_gemini_schema(schema: dict) -> Any:
    """JSON Schema -> the SDK's Schema, keeping only what it accepts."""
    from google.genai import types

    type_map = {
        "object": "OBJECT",
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
    }

    def convert(node: dict) -> Any:
        kind = type_map.get(node.get("type", "string"), "STRING")
        kwargs: dict[str, Any] = {"type": kind}
        if node.get("description"):
            kwargs["description"] = node["description"]
        if kind == "OBJECT":
            properties = {key: convert(value) for key, value in (node.get("properties") or {}).items()}
            if properties:
                kwargs["properties"] = properties
            if node.get("required"):
                kwargs["required"] = list(node["required"])
        if kind == "ARRAY":
            kwargs["items"] = convert(node.get("items") or {"type": "string"})
        return types.Schema(**kwargs)

    return convert(schema)


def _to_gemini_content(turn: Turn) -> Any:
    from google.genai import types

    if turn.role == "tool":
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=turn.tool_name or "tool",
                    response=turn.tool_result or {},
                )
            ],
        )

    if turn.role == "assistant":
        parts = []
        if turn.text:
            parts.append(types.Part(text=turn.text))
        for call in turn.tool_calls:
            part = types.Part.from_function_call(name=call.name, args=call.arguments)
            if call.thought_signature:
                part.thought_signature = call.thought_signature
            parts.append(part)
        return types.Content(role="model", parts=parts or [types.Part(text=" ")])

    return types.Content(role="user", parts=[types.Part(text=turn.text or " ")])


def _from_gemini_response(response) -> LLMReply:
    """Walk the parts. `response.text` raises when a reply is all function calls."""
    texts: list[str] = []
    calls: list[ToolCall] = []
    finish = None

    for candidate in getattr(response, "candidates", None) or []:
        finish = getattr(candidate, "finish_reason", None) or finish
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                texts.append(part.text)
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                calls.append(
                    ToolCall(
                        name=call.name,
                        arguments=dict(getattr(call, "args", None) or {}),
                        thought_signature=getattr(part, "thought_signature", None),
                    )
                )

    usage = getattr(response, "usage_metadata", None)
    return LLMReply(
        text="\n".join(t.strip() for t in texts if t and t.strip()),
        tool_calls=calls,
        input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        finish_reason=str(finish) if finish else None,
    )


# --- the fake ---------------------------------------------------------------


class FakeProvider:
    """A scripted provider for tests.

    Takes a list of replies and returns them in order, recording what it was
    asked. Every test of the agent loop, the tool dispatch and the token budget
    runs against this — no key, no network, no flake.
    """

    name = "fake"

    def __init__(self, replies: list[LLMReply], model: str = "fake-1"):
        self.model = model
        self._replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system: str, turns: list[Turn], tools: list[Any]) -> LLMReply:
        self.calls.append({"system": system, "turns": list(turns), "tools": [t.name for t in tools]})
        if not self._replies:
            return LLMReply(text="(the fake provider ran out of scripted replies)")
        return self._replies.pop(0)


# --- selection --------------------------------------------------------------

#: Set by tests to force a provider without touching settings.
_override: LLMProvider | None = None


def set_provider_override(provider: LLMProvider | None) -> None:
    global _override
    _override = provider


def get_provider() -> LLMProvider | None:
    """The configured provider, or None when the app should degrade gracefully."""
    if _override is not None:
        return _override
    api_key = settings.gemini_key
    if not api_key:
        return None
    try:
        return GeminiProvider(api_key=api_key, model=settings.gemini_model)
    except Exception as exc:
        logger.warning("Could not construct the Gemini provider: %s", exc)
        return None


def estimate_tokens(text: str) -> int:
    """A rough token count for budgeting, without an SDK round-trip.

    Deliberately approximate — it decides when to trim history, not what to
    bill. Four characters per token is close enough for English prose and
    conservative for the JSON that tool results are made of.
    """
    return max(1, len(text) // 4)


def serialise_for_budget(turn: Turn) -> int:
    payload = turn.text or ""
    if turn.tool_result is not None:
        payload += json.dumps(turn.tool_result, default=str)
    for call in turn.tool_calls:
        payload += call.name + json.dumps(call.arguments, default=str)
    return estimate_tokens(payload)
