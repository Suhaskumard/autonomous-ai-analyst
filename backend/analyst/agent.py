"""The loop: ask, run the tools it asks for, ask again, answer.

Bounded on purpose. `agent_max_steps` caps how many tool calls one question
may take, so a model that keeps deciding it needs one more correlation cannot
bill indefinitely — when the budget runs out it is told to answer with what it
already has, rather than being cut off mid-thought.

Every step is recorded. The UI renders them as the work behind the answer, and
that record is also what makes the answer checkable: the code that ran, the
table it returned, the figure it drew.
"""

import json
import logging
from dataclasses import dataclass, field

from analyst.providers import LLMProvider, LLMReply, Turn, serialise_for_budget
from analyst.tools import Tool, execute_tool
from config import settings

logger = logging.getLogger(__name__)

# Tool output the model reads back is capped: a 200-row table would eat the
# context window for a question the summary already answers.
MAX_TOOL_SUMMARY_CHARS = 4000


@dataclass
class AgentStep:
    """One tool call and what it produced."""

    tool: str
    arguments: dict
    ok: bool
    summary: str
    payload: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "summary": self.summary,
            **self.payload,
        }


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stopped_at_step_limit: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "steps": [step.as_dict() for step in self.steps],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stopped_at_step_limit": self.stopped_at_step_limit,
        }


# Keys that belong to the UI but would be noise in the model's context.
_UI_ONLY_KEYS = ("figures", "table", "matrix", "execution", "top_values", "by_segment")


def run_agent(
    provider: LLMProvider,
    system_prompt: str,
    turns: list[Turn],
    toolset: dict[str, Tool],
    max_steps: int | None = None,
) -> AgentResult:
    """Drive provider and tools until there is an answer or the budget is spent."""
    max_steps = int(max_steps or settings.agent_max_steps)
    tools = list(toolset.values())
    working: list[Turn] = list(turns)
    steps: list[AgentStep] = []
    input_tokens = output_tokens = 0

    for step_number in range(max_steps + 1):
        last_chance = step_number == max_steps
        available = [] if last_chance else tools

        if last_chance:
            working.append(
                Turn(
                    role="user",
                    text=(
                        "You have used the tool budget for this question. Answer now using the results "
                        "you already have, and say plainly what you could not check."
                    ),
                )
            )

        reply: LLMReply = provider.complete(system_prompt, working, available)
        input_tokens += reply.input_tokens
        output_tokens += reply.output_tokens

        # On the last pass the model was offered no tools, so any call it makes
        # anyway is ignored: the budget is a ceiling, not a suggestion.
        if last_chance or not reply.tool_calls:
            answer = reply.text.strip() or _empty_answer(reply, steps)
            return AgentResult(
                answer=answer,
                steps=steps,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stopped_at_step_limit=last_chance and bool(steps),
            )

        working.append(Turn(role="assistant", text=reply.text, tool_calls=reply.tool_calls))

        for call in reply.tool_calls:
            # `args` is reserved on LogRecord; naming it that raises inside logging.
            logger.info("Analyst tool call", extra={"tool": call.name, "tool_args": list(call.arguments)})
            result = execute_tool(toolset, call.name, call.arguments)
            step = AgentStep(
                tool=call.name,
                arguments=call.arguments,
                ok=bool(result.get("ok")),
                summary=str(result.get("summary", ""))[:MAX_TOOL_SUMMARY_CHARS],
                payload={key: value for key, value in result.items() if key not in {"ok", "summary", "error"}},
            )
            steps.append(step)
            working.append(
                Turn(
                    role="tool",
                    tool_name=call.name,
                    tool_result=_for_model(result),
                )
            )

    # Unreachable: the last iteration always returns.
    return AgentResult(answer="The analyst could not complete this question.", steps=steps)


def _for_model(result: dict) -> dict:
    """The compact view of a tool result that goes back into the context.

    Images and full tables are for the user, not the model — it already has
    the summary, and re-reading 200 rows of JSON buys nothing.
    """
    trimmed = {key: value for key, value in result.items() if key not in _UI_ONLY_KEYS}
    payload = json.dumps(trimmed, default=str)
    if len(payload) > MAX_TOOL_SUMMARY_CHARS:
        return {"ok": result.get("ok", False), "summary": str(result.get("summary", ""))[:MAX_TOOL_SUMMARY_CHARS]}
    return trimmed


def _empty_answer(reply: LLMReply, steps: list[AgentStep]) -> str:
    if steps:
        return "I ran the analysis below but did not produce a written summary. The results are shown above."
    if reply.finish_reason:
        return f"The model returned no readable content (finish reason: {reply.finish_reason})."
    return "The model returned an empty response. Try rephrasing the question."


def trim_to_budget(turns: list[Turn], budget_tokens: int) -> tuple[list[Turn], bool]:
    """Drop the oldest turns until the conversation fits the budget.

    Long sessions used to silently lose their earliest context inside the
    provider; trimming here is explicit, and the caller tells the user it
    happened rather than letting the answer quietly drift.
    """
    if budget_tokens <= 0:
        return turns, False

    total = sum(serialise_for_budget(turn) for turn in turns)
    if total <= budget_tokens:
        return turns, False

    kept: list[Turn] = []
    running = 0
    # Newest first: the recent turns are the ones the question depends on.
    for turn in reversed(turns):
        cost = serialise_for_budget(turn)
        if running + cost > budget_tokens and kept:
            break
        kept.append(turn)
        running += cost

    kept.reverse()
    # A conversation must not begin with a tool result whose call was dropped.
    while kept and kept[0].role == "tool":
        kept.pop(0)
    return kept, True
