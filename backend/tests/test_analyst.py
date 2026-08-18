"""The sandbox, the tools, the agent loop, the guards, and the report.

Phase 5's exit criterion is that "which two features correlate most strongly,
and does that hold within each segment?" is answered by executed code whose
output you can verify by hand. `test_the_exit_criterion_question` does exactly
that: it runs the question through the agent and then recomputes every number
independently with pandas.

Every LLM-dependent test runs against FakeProvider, so the suite needs no API
key and never touches the network.
"""

import pytest

import db
from analyst import report as report_module
from analyst import sandbox
from analyst.agent import AgentStep, run_agent, trim_to_budget
from analyst.providers import FakeProvider, LLMReply, ToolCall, Turn, estimate_tokens, set_provider_override
from analyst.tools import build_toolset, execute_tool
from config import settings
from tests.conftest import upload_and_wait

FIXTURE = "clean_classification.csv"


@pytest.fixture
def frame(fixture_df):
    return fixture_df(FIXTURE)


@pytest.fixture
def snapshot(fixture_csv):
    return fixture_csv(FIXTURE)


@pytest.fixture
def toolset(frame, snapshot):
    metadata = {"target": "churn", "problem_type": "classification", "selected_model": "RandomForest"}
    return build_toolset(frame, metadata, snapshot, "a" * 64)


@pytest.fixture
def fake_provider():
    """Install a scripted provider for the duration of a test."""

    def _install(replies):
        provider = FakeProvider(replies)
        set_provider_override(provider)
        return provider

    yield _install
    set_provider_override(None)


# --- the sandbox ------------------------------------------------------------


def test_code_runs_against_the_real_rows(snapshot):
    result = sandbox.run_code("len(df)", snapshot)
    assert result.ok, result.error
    assert result.result == {"kind": "scalar", "value": 240}


def test_printed_output_is_captured(snapshot):
    result = sandbox.run_code("print('hello'); print(2 + 2)", snapshot)
    assert result.ok
    assert "hello" in result.stdout
    assert "4" in result.stdout


def test_a_dataframe_result_comes_back_as_a_table(snapshot):
    result = sandbox.run_code("df.head(3)", snapshot)
    assert result.result["kind"] == "table"
    assert result.result["shape"][0] == 3
    assert "age" in result.result["columns"]


def test_a_plot_comes_back_as_an_image(snapshot):
    result = sandbox.run_code("plt.figure(); df['age'].hist()", snapshot)
    assert result.ok, result.error
    assert len(result.figures) == 1
    assert result.figures[0]["mime"] == "image/png"
    assert len(result.figures[0]["data"]) > 500


def test_an_error_is_reported_rather_than_raised(snapshot):
    result = sandbox.run_code("df['nope'].mean()", snapshot)
    assert not result.ok
    assert "nope" in result.error


def test_an_infinite_loop_is_killed(snapshot):
    result = sandbox.run_code("while True: pass", snapshot, timeout_seconds=3)
    assert not result.ok
    assert "timed out" in result.error


@pytest.mark.parametrize(
    "code,expected",
    [
        ("import socket", "socket"),
        ("import subprocess", "subprocess"),
        ("import ctypes", "ctypes"),
    ],
)
def test_escape_hatch_imports_are_refused(snapshot, code, expected):
    result = sandbox.run_code(code, snapshot)
    assert not result.ok
    assert expected in result.error


def test_writing_a_file_is_refused(snapshot):
    result = sandbox.run_code("open('escaped.txt', 'w').write('x')", snapshot)
    assert not result.ok
    assert "Writing files" in result.error


def test_os_system_is_refused(snapshot):
    result = sandbox.run_code("import os\nos.system('echo hi')", snapshot)
    assert not result.ok
    assert "os.system" in result.error


def test_scientific_libraries_still_work(snapshot):
    """The guards must not brick the libraries the analyst exists to use."""
    result = sandbox.run_code(
        "from scipy import stats\nround(float(stats.pearsonr(df['age'], df['spend'])[0]), 4)", snapshot
    )
    assert result.ok, result.error
    assert isinstance(result.result["value"], float)


def test_the_sandbox_is_not_given_the_api_key(snapshot, monkeypatch):
    """Analysis code must not be able to read the operator's credentials."""
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-value")
    result = sandbox.run_code("import os\nos.environ.get('GEMINI_API_KEY', 'ABSENT')", snapshot)
    assert result.ok, result.error
    assert result.result["value"] == "ABSENT"


def test_oversized_code_is_refused_without_spawning(snapshot):
    result = sandbox.run_code("x = 1\n" * 20_000, snapshot)
    assert not result.ok
    assert "limit" in result.error


# --- the tools --------------------------------------------------------------


def test_describe_column_reports_real_statistics(toolset, frame):
    result = execute_tool(toolset, "describe_column", {"column": "age"})
    assert result["ok"]
    assert result["stats"]["mean"] == pytest.approx(round(frame["age"].mean(), 6))
    assert result["distinct"] == frame["age"].nunique()


def test_describe_column_names_the_columns_that_exist(toolset):
    result = execute_tool(toolset, "describe_column", {"column": "nope"})
    assert not result["ok"]
    assert "age" in result["summary"]


def test_correlate_ranks_pairs_by_absolute_strength(toolset, frame):
    result = execute_tool(toolset, "correlate", {"top": 3})
    assert result["ok"]
    strengths = [abs(p["r"]) for p in result["top_pairs"]]
    assert strengths == sorted(strengths, reverse=True)


def test_filter_rows_counts_and_samples(toolset, frame):
    result = execute_tool(toolset, "filter_rows", {"expression": "age > 50", "limit": 5})
    assert result["ok"]
    assert result["matched"] == int((frame["age"] > 50).sum())
    assert len(result["table"]["rows"]) <= 5


def test_filter_rows_reports_a_bad_expression(toolset):
    result = execute_tool(toolset, "filter_rows", {"expression": "age >>> 50"})
    assert not result["ok"]


def test_group_stats_matches_pandas(toolset, frame):
    result = execute_tool(toolset, "group_stats", {"by": "region", "column": "spend", "agg": "mean"})
    assert result["ok"]
    expected = frame.groupby("region")["spend"].mean().round(6).to_dict()
    for region, value in result["table"]["rows"]:
        assert value == pytest.approx(expected[region], abs=1e-4)


def test_group_stats_refuses_a_nonsense_aggregate(toolset):
    assert not execute_tool(toolset, "group_stats", {"by": "region", "column": "spend", "agg": "sqrt"})["ok"]


def test_plot_returns_an_image(toolset):
    result = execute_tool(toolset, "plot", {"kind": "hist", "x": "age"})
    assert result["ok"], result.get("summary")
    assert result["figures"] and result["figures"][0]["mime"] == "image/png"


def test_plot_requires_y_for_a_scatter(toolset):
    assert not execute_tool(toolset, "plot", {"kind": "scatter", "x": "age"})["ok"]


def test_run_python_is_the_escape_hatch(toolset):
    result = execute_tool(toolset, "run_python", {"code": "int(df['age'].max())"})
    assert result["ok"]
    assert result["execution"]["result"]["value"] == 79


def test_an_unknown_tool_is_reported_not_raised(toolset):
    result = execute_tool(toolset, "teleport", {})
    assert not result["ok"]
    assert "no tool called" in result["summary"]


def test_wrong_arguments_are_reported_not_raised(toolset):
    result = execute_tool(toolset, "describe_column", {"colunm": "age"})
    assert not result["ok"]


# --- the agent loop ---------------------------------------------------------


def test_the_agent_runs_the_tool_it_asks_for(toolset):
    provider = FakeProvider(
        [
            LLMReply(tool_calls=[ToolCall("describe_column", {"column": "age"})]),
            LLMReply(text="The mean age is 48.85."),
        ]
    )
    result = run_agent(provider, "sys", [Turn(role="user", text="describe age")], toolset)

    assert result.answer == "The mean age is 48.85."
    assert [step.tool for step in result.steps] == ["describe_column"]
    assert result.steps[0].ok


def test_the_agent_sums_tokens_across_the_loop(toolset):
    provider = FakeProvider(
        [
            LLMReply(tool_calls=[ToolCall("describe_column", {"column": "age"})], input_tokens=100, output_tokens=10),
            LLMReply(text="done", input_tokens=150, output_tokens=25),
        ]
    )
    result = run_agent(provider, "sys", [Turn(role="user", text="q")], toolset)
    assert (result.input_tokens, result.output_tokens) == (250, 35)
    assert result.total_tokens == 285


def test_the_step_budget_stops_a_runaway_loop(toolset):
    """A model that only ever asks for another tool must still terminate."""
    provider = FakeProvider([LLMReply(tool_calls=[ToolCall("describe_column", {"column": "age"})])] * 20)
    result = run_agent(provider, "sys", [Turn(role="user", text="q")], toolset, max_steps=3)

    assert len(result.steps) == 3
    assert result.stopped_at_step_limit
    # The last call is made with no tools, so the model has to answer.
    assert provider.calls[-1]["tools"] == []


def test_a_failing_tool_is_fed_back_rather_than_crashing(toolset):
    provider = FakeProvider(
        [
            LLMReply(tool_calls=[ToolCall("describe_column", {"column": "missing"})]),
            LLMReply(text="That column does not exist."),
        ]
    )
    result = run_agent(provider, "sys", [Turn(role="user", text="q")], toolset)
    assert not result.steps[0].ok
    assert result.answer == "That column does not exist."


def test_an_empty_reply_does_not_surface_as_a_blank_answer(toolset):
    provider = FakeProvider([LLMReply(text="", finish_reason="SAFETY")])
    result = run_agent(provider, "sys", [Turn(role="user", text="q")], toolset)
    assert "SAFETY" in result.answer


def test_images_and_tables_are_kept_out_of_the_model_context(toolset):
    """The user sees the figure; re-feeding base64 PNG to the model is waste."""
    provider = FakeProvider(
        [LLMReply(tool_calls=[ToolCall("plot", {"kind": "hist", "x": "age"})]), LLMReply(text="Drawn.")]
    )
    result = run_agent(provider, "sys", [Turn(role="user", text="plot age")], toolset)

    assert result.steps[0].payload["figures"], "the UI still gets the image"
    tool_turn = next(t for t in provider.calls[-1]["turns"] if t.role == "tool")
    assert "figures" not in (tool_turn.tool_result or {})


# --- token budgeting --------------------------------------------------------


def test_a_short_conversation_is_left_alone():
    turns = [Turn(role="user", text="hello")]
    kept, trimmed = trim_to_budget(turns, budget_tokens=1000)
    assert kept == turns and not trimmed


def test_a_long_conversation_drops_its_oldest_turns():
    turns = [Turn(role="user", text="x" * 4000) for _ in range(10)]
    kept, trimmed = trim_to_budget(turns, budget_tokens=1500)

    assert trimmed
    assert 0 < len(kept) < len(turns)
    # The newest turn survives — it is the one the question depends on.
    assert kept[-1] is turns[-1]


def test_trimming_never_starts_a_conversation_with_an_orphan_tool_result():
    turns = [
        Turn(role="user", text="y" * 8000),
        Turn(role="assistant", text="a", tool_calls=[ToolCall("describe_column", {"column": "age"})]),
        Turn(role="tool", tool_name="describe_column", tool_result={"ok": True}),
        Turn(role="user", text="follow up"),
    ]
    kept, _ = trim_to_budget(turns, budget_tokens=30)
    assert not kept or kept[0].role != "tool"


def test_token_estimate_grows_with_length():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40) > 0


# --- the exit criterion -----------------------------------------------------


def test_the_exit_criterion_question(toolset, frame):
    """ "Which two features correlate most strongly, and does that hold within
    each segment?" — answered by executed code, verified here by hand."""
    provider = FakeProvider(
        [
            LLMReply(tool_calls=[ToolCall("correlate", {"within": "region", "top": 3})]),
            LLMReply(text="age and spend correlate most strongly, and it does not hold in every region."),
        ]
    )
    result = run_agent(
        provider,
        "sys",
        [Turn(role="user", text="Which two features correlate most strongly, and does it hold per segment?")],
        toolset,
    )

    step = result.steps[0]
    assert step.ok

    # The headline pair is the strongest pair in the data.
    headline = step.payload["headline_pair"]
    assert {headline["a"], headline["b"]} == {"age", "spend"}
    assert headline["r"] == pytest.approx(round(frame["age"].corr(frame["spend"]), 4))

    # And every per-segment number recomputes from the CSV.
    for row in step.payload["by_segment"]:
        subset = frame[frame["region"].astype(str) == row["segment"]]
        assert row["n"] == len(subset)
        assert row["r"] == pytest.approx(round(subset["age"].corr(subset["spend"]), 4))


# --- the endpoint -----------------------------------------------------------


def test_chat_answers_and_persists_the_conversation(client, fixture_csv, fake_provider):
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    fake_provider(
        [
            LLMReply(tool_calls=[ToolCall("describe_column", {"column": "age"})]),
            LLMReply(text="Mean age is 48.85.", input_tokens=120, output_tokens=30),
        ]
    )

    body = client.post("/api/chat", json={"run_key": run_key, "query": "describe age"}).json()

    assert body["response"]["answer"] == "Mean age is 48.85."
    assert body["response"]["data_access"] == "local-execution"
    assert body["response"]["steps"][0]["tool"] == "describe_column"
    assert body["conversation_id"]

    stored = client.get(f"/api/chat/conversation/{body['conversation_id']}").json()
    assert [m["role"] for m in stored["messages"]] == ["user", "assistant"]
    assert stored["messages"][0]["content"] == "describe age"


def test_history_comes_from_the_server_not_the_client(client, fixture_csv, fake_provider):
    """The client posts a question and an id, never a transcript."""
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    provider = fake_provider([LLMReply(text="one"), LLMReply(text="two")])

    first = client.post("/api/chat", json={"run_key": run_key, "query": "first question"}).json()
    conversation_id = first["conversation_id"]
    client.post(
        "/api/chat",
        json={"run_key": run_key, "query": "second question", "conversation_id": conversation_id},
    )

    # The second call's context was rebuilt from stored messages.
    texts = [turn.text for turn in provider.calls[-1]["turns"]]
    assert "first question" in texts
    assert "one" in texts
    assert "second question" in texts


def test_a_conversation_cannot_be_borrowed_by_another_run(client, fixture_csv, fake_provider):
    fake_provider([LLMReply(text="ok")])
    first = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    second = upload_and_wait(client, fixture_csv("clean_regression.csv"))["result"]["run_key"]

    started = client.post("/api/chat", json={"run_key": first, "query": "q"}).json()
    response = client.post(
        "/api/chat",
        json={"run_key": second, "query": "q", "conversation_id": started["conversation_id"]},
    )
    assert response.status_code == 400


def test_an_unknown_conversation_is_404(client, fixture_csv, fake_provider):
    fake_provider([LLMReply(text="ok")])
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    response = client.post("/api/chat", json={"run_key": run_key, "query": "q", "conversation_id": "does-not-exist"})
    assert response.status_code == 404


def test_the_message_rate_limit_is_enforced(client, fixture_csv, fake_provider, monkeypatch):
    monkeypatch.setattr(settings, "chat_max_messages_per_hour", 2)
    fake_provider([LLMReply(text="ok")] * 6)
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]

    first = client.post("/api/chat", json={"run_key": run_key, "query": "one"}).json()
    conversation_id = first["conversation_id"]
    client.post("/api/chat", json={"run_key": run_key, "query": "two", "conversation_id": conversation_id})
    third = client.post("/api/chat", json={"run_key": run_key, "query": "three", "conversation_id": conversation_id})

    assert third.status_code == 429
    assert "hourly limit" in third.json()["detail"]


def test_the_token_budget_is_enforced(client, fixture_csv, fake_provider, monkeypatch):
    monkeypatch.setattr(settings, "chat_token_budget", 100)
    fake_provider([LLMReply(text="ok", input_tokens=200, output_tokens=50)] * 4)
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]

    first = client.post("/api/chat", json={"run_key": run_key, "query": "one"}).json()
    second = client.post(
        "/api/chat",
        json={"run_key": run_key, "query": "two", "conversation_id": first["conversation_id"]},
    )

    assert second.status_code == 429
    assert "token budget" in second.json()["detail"]


def test_chat_degrades_gracefully_without_a_provider(client, fixture_csv):
    """No key must mean an honest answer, not a stack trace."""
    set_provider_override(None)
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]

    body = client.post("/api/chat", json={"run_key": run_key, "query": "hello"}).json()

    assert body["response"]["llm_enabled"] is False
    assert "GEMINI_API_KEY" in body["response"]["answer"]
    # The exchange is still recorded, so the transcript is not silently lossy.
    assert len(client.get(f"/api/chat/conversation/{body['conversation_id']}").json()["messages"]) == 2


def test_an_empty_question_is_rejected(client, fixture_csv, fake_provider):
    fake_provider([LLMReply(text="ok")])
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    assert client.post("/api/chat", json={"run_key": run_key, "query": "   "}).status_code == 422


def test_an_oversized_question_is_rejected(client, fixture_csv, fake_provider):
    fake_provider([LLMReply(text="ok")])
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    response = client.post("/api/chat", json={"run_key": run_key, "query": "x" * 5000})
    assert response.status_code == 413


def test_chat_rejects_a_malformed_run_key(client, fake_provider):
    fake_provider([LLMReply(text="ok")])
    assert client.post("/api/chat", json={"run_key": "../etc/passwd", "query": "q"}).status_code == 400


def test_conversations_are_listed_per_run(client, fixture_csv, fake_provider):
    fake_provider([LLMReply(text="ok")] * 4)
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    client.post("/api/chat", json={"run_key": run_key, "query": "first ever question"})

    listed = client.get(f"/api/chat/{run_key}/conversations").json()
    assert len(listed["conversations"]) == 1
    # The opening question is the session's label.
    assert listed["conversations"][0]["title"] == "first ever question"


def test_deleting_a_run_deletes_its_conversations(client, fixture_csv, fake_provider):
    fake_provider([LLMReply(text="ok")] * 2)
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]
    started = client.post("/api/chat", json={"run_key": run_key, "query": "q"}).json()

    body = client.delete(f"/api/runs/{run_key}").json()

    assert body["conversations_removed"] == 1
    assert db.get_conversation(started["conversation_id"]) is None


# --- the report -------------------------------------------------------------


def test_the_report_is_built_from_computed_findings(frame, snapshot, fake_provider):
    provider = FakeProvider([LLMReply(text="A written summary.")])
    metadata = {
        "target": "churn",
        "problem_type": "classification",
        "selected_model": "RandomForest",
        "selection_metric": "f1_macro",
        "health": {"verdict": "strong", "headline": "Model beats the naive baseline"},
    }

    report = report_module.build_report(frame, metadata, snapshot, "a" * 64, provider=provider)

    sections = {section["id"] for section in report["sections"]}
    assert {"overview", "quality", "columns", "correlations", "model"} <= sections
    assert report["narrative"] == "A written summary."
    assert report["dataset"]["rows"] == len(frame)


def test_the_report_answers_the_segment_question_unprompted(frame, snapshot):
    report = report_module.build_report(frame, {"target": "churn"}, snapshot, "a" * 64, provider=None)
    segments = next((s for s in report["sections"] if s["id"] == "segments"), None)

    assert segments is not None
    for row in segments["facts"]["by_segment"]:
        subset = frame[frame["region"].astype(str) == row["segment"]]
        assert row["n"] == len(subset)


def test_the_report_still_builds_without_a_model(frame, snapshot):
    """Findings without narrative beat narrative without findings."""
    report = report_module.build_report(frame, {"target": "churn"}, snapshot, "a" * 64, provider=None)

    assert report["narrative"] == ""
    assert "No language model is configured" in report["narrative_error"]
    assert any(section["id"] == "correlations" for section in report["sections"])


def test_a_failing_narrative_does_not_lose_the_findings(frame, snapshot):
    class Exploding:
        name, model = "boom", "boom-1"

        def complete(self, *args, **kwargs):
            raise RuntimeError("quota exhausted")

    report = report_module.build_report(frame, {"target": "churn"}, snapshot, "a" * 64, provider=Exploding())

    assert "quota exhausted" in report["narrative_error"]
    assert any(section["id"] == "correlations" for section in report["sections"])


def test_the_report_renders_as_shareable_markdown(frame, snapshot):
    report = report_module.build_report(frame, {"target": "churn"}, snapshot, "a" * 64, provider=None)
    markdown = report_module.render_markdown(report)

    assert markdown.startswith("# Analysis report")
    assert "## Overview" in markdown
    assert "## Relationships" in markdown


def test_the_report_endpoint_returns_markdown(client, fixture_csv, fake_provider):
    fake_provider([LLMReply(text="Narrative.")])
    run_key = upload_and_wait(client, fixture_csv(FIXTURE))["result"]["run_key"]

    body = client.post(f"/api/report/{run_key}").json()

    assert body["narrative"] == "Narrative."
    assert body["markdown"].startswith("# Analysis report")
    assert body["run_key"] == run_key


def test_the_report_endpoint_rejects_a_malformed_key(client):
    assert client.post("/api/report/not-a-key").status_code == 400


# --- provider abstraction ---------------------------------------------------


def test_the_fake_provider_records_what_it_was_asked(toolset):
    provider = FakeProvider([LLMReply(text="hi")])
    run_agent(provider, "the system prompt", [Turn(role="user", text="q")], toolset)

    assert provider.calls[0]["system"] == "the system prompt"
    assert "correlate" in provider.calls[0]["tools"]


def test_a_provider_that_runs_out_of_replies_says_so(toolset):
    provider = FakeProvider([])
    result = run_agent(provider, "sys", [Turn(role="user", text="q")], toolset)
    assert "ran out of scripted replies" in result.answer


def test_step_serialisation_keeps_the_payload_for_the_ui():
    step = AgentStep(tool="plot", arguments={"kind": "hist"}, ok=True, summary="drew it", payload={"figures": [1]})
    assert step.as_dict()["figures"] == [1]
    assert step.as_dict()["tool"] == "plot"
