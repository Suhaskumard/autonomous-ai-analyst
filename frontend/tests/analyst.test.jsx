import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../services/api", () => ({
  startUploadJob: vi.fn(),
  getUploadJobStatus: vi.fn(),
  profileDataset: vi.fn(),
  predictDataset: vi.fn(),
  predictSingleRow: vi.fn(),
  chatQuery: vi.fn(),
  listRuns: vi.fn(),
  getRunResult: vi.fn(),
  compareRuns: vi.fn(),
  deleteRun: vi.fn(),
  getInsights: vi.fn(),
  generateReport: vi.fn(),
  listConversations: vi.fn(),
  getConversation: vi.fn(),
  whoAmI: vi.fn(() => Promise.resolve({ user_id: "local", email: "local@localhost", auth_enabled: false })),
  getUsage: vi.fn(() => Promise.resolve({ calls: 0, estimated_cost_usd: 0, window_days: 30 })),
  getPrivacyPolicy: vi.fn(),
  deleteAllMyData: vi.fn(),
  getApiKey: vi.fn(() => ""),
  setApiKey: vi.fn(),
  clearApiKey: vi.fn(),
  UnauthorizedError: class UnauthorizedError extends Error {},
}));

import ChatBox from "../components/ChatBox";
import ReportPanel from "../components/ReportPanel";
import * as api from "../services/api";

const RUN_KEY = "a".repeat(64);

// A 1x1 PNG, enough to prove an <img> is rendered from base64.
const PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

function reply(overrides = {}) {
  return {
    run_key: RUN_KEY,
    conversation_id: "conv-1",
    query: "q",
    response: {
      answer: "The strongest pair is **age** and **spend** (r = 0.13).",
      steps: [],
      llm_enabled: true,
      llm_provider: "gemini",
      llm_model: "gemini-1.5-flash-8b",
      data_access: "local-execution",
      ...overrides,
    },
  };
}

const CORRELATE_STEP = {
  tool: "correlate",
  arguments: { within: "region" },
  ok: true,
  summary: "Strongest correlations overall: age vs spend r=0.1302",
  matrix: {
    kind: "table",
    columns: ["column", "age", "spend"],
    rows: [
      ["age", 1, 0.1302],
      ["spend", 0.1302, 1],
    ],
    shape: [2, 3],
    truncated: false,
  },
};

const PYTHON_STEP = {
  tool: "run_python",
  arguments: { code: "df['age'].mean()" },
  ok: true,
  summary: "result: 48.85",
  execution: {
    ok: true,
    code: "df['age'].mean()",
    stdout: "computed\n",
    result: { kind: "scalar", value: 48.85 },
    figures: [],
  },
};

const PLOT_STEP = {
  tool: "plot",
  arguments: { kind: "hist", x: "age" },
  ok: true,
  summary: "Drew a hist plot of age.",
  figures: [{ kind: "image", mime: "image/png", data: PNG, caption: "hist of age" }],
};

describe("ChatBox", () => {
  beforeEach(() => vi.clearAllMocks());

  async function ask(user, text = "which two features correlate most strongly?") {
    await user.type(screen.getByLabelText(/Ask the analyst/i), text);
    await user.click(screen.getByRole("button", { name: /Send/i }));
  }

  it("sends only a question and a conversation id, never a transcript", async () => {
    api.chatQuery.mockResolvedValue(reply());
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user, "first");
    await waitFor(() => expect(api.chatQuery).toHaveBeenCalled());

    const payload = api.chatQuery.mock.calls[0][0];
    expect(payload).toEqual({ run_key: RUN_KEY, query: "first", conversation_id: null });
    // The old client posted its own history back; nothing here does.
    expect(payload).not.toHaveProperty("history");
  });

  it("reuses the server's conversation id on the next question", async () => {
    api.chatQuery.mockResolvedValue(reply());
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user, "first");
    await screen.findByText(/strongest pair/i);
    await ask(user, "second");

    await waitFor(() => expect(api.chatQuery).toHaveBeenCalledTimes(2));
    expect(api.chatQuery.mock.calls[1][0].conversation_id).toBe("conv-1");
  });

  it("renders a result table rather than flattening it to text", async () => {
    api.chatQuery.mockResolvedValue(reply({ steps: [CORRELATE_STEP] }));
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user);
    await user.click(await screen.findByRole("button", { name: /Computed correlations/i }));

    const table = await screen.findByRole("table");
    // A correlation matrix is symmetric, so the value appears on both sides.
    expect(within(table).getAllByText("0.1302")).toHaveLength(2);
    expect(within(table).getByText("column")).toBeInTheDocument();
  });

  it("shows the code behind a number so the answer can be checked", async () => {
    api.chatQuery.mockResolvedValue(reply({ steps: [PYTHON_STEP] }));
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user);
    await user.click(await screen.findByRole("button", { name: /Executed Python/i }));

    expect(screen.getByText("df['age'].mean()")).toBeInTheDocument();
    expect(screen.getByText(/computed/)).toBeInTheDocument();
  });

  it("renders a chart the analyst drew as an image", async () => {
    api.chatQuery.mockResolvedValue(reply({ steps: [PLOT_STEP] }));
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user);
    await user.click(await screen.findByRole("button", { name: /Drew a chart/i }));

    const image = await screen.findByRole("img", { name: /hist of age/i });
    expect(image).toHaveAttribute("src", `data:image/png;base64,${PNG}`);
  });

  it("marks a failed step instead of hiding it", async () => {
    api.chatQuery.mockResolvedValue(
      reply({
        steps: [{ tool: "describe_column", arguments: { column: "nope" }, ok: false, summary: "No column 'nope'." }],
      })
    );
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user);
    expect(await screen.findByText(/No column 'nope'/)).toBeInTheDocument();
  });

  it("reports the model and that execution is local", async () => {
    api.chatQuery.mockResolvedValue(reply());
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user);

    expect(await screen.findByText("gemini-1.5-flash-8b")).toBeInTheDocument();
    expect(screen.getByText(/Runs code locally on your rows/i)).toBeInTheDocument();
  });

  it("says plainly when no model is configured", async () => {
    api.chatQuery.mockResolvedValue(
      reply({ llm_enabled: false, llm_model: undefined, answer: "No language model is configured." })
    );
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user);
    expect(await screen.findByText(/Disabled \(no API key\)/i)).toBeInTheDocument();
  });

  it("restores the question when the request fails, so it is not lost", async () => {
    api.chatQuery.mockRejectedValue(new Error("Rate limit reached."));
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user, "an expensive question");

    expect(await screen.findByText("Rate limit reached.")).toBeInTheDocument();
    expect(screen.getByLabelText(/Ask the analyst/i)).toHaveValue("an expensive question");
  });

  it("starts a fresh conversation on request", async () => {
    api.chatQuery.mockResolvedValue(reply());
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await ask(user, "first");
    await screen.findByText(/strongest pair/i);
    await user.click(screen.getByRole("button", { name: /New conversation/i }));
    await ask(user, "second");

    await waitFor(() => expect(api.chatQuery).toHaveBeenCalledTimes(2));
    expect(api.chatQuery.mock.calls[1][0].conversation_id).toBeNull();
  });

  it("offers the exit-criterion question as a starting point", async () => {
    api.chatQuery.mockResolvedValue(reply());
    const user = userEvent.setup();
    render(<ChatBox runKey={RUN_KEY} onAsk={api.chatQuery} />);

    await user.click(screen.getByRole("button", { name: /correlate most strongly.*within each segment/i }));

    await waitFor(() => expect(api.chatQuery).toHaveBeenCalled());
    expect(api.chatQuery.mock.calls[0][0].query).toMatch(/within each segment/i);
  });
});

describe("ReportPanel", () => {
  beforeEach(() => vi.clearAllMocks());

  const REPORT = {
    run_key: RUN_KEY,
    dataset: { rows: 240, columns: 4, target: "churn", problem_type: "classification" },
    narrative: "The dataset is small but clean.",
    narrative_error: null,
    sections: [
      { id: "overview", title: "Overview", summary: "240 rows and 4 columns." },
      {
        id: "charts",
        title: "Distributions",
        summary: "1 chart drawn.",
        figures: [{ mime: "image/png", data: PNG, caption: "hist of age" }],
      },
    ],
    markdown: "# Analysis report — churn\n\n## Overview",
  };

  it("generates a report on request and renders its sections", async () => {
    api.generateReport.mockResolvedValue(REPORT);
    const user = userEvent.setup();
    render(<ReportPanel runKey={RUN_KEY} onGenerate={api.generateReport} />);

    await user.click(screen.getByRole("button", { name: /Generate report/i }));

    expect(await screen.findByText("The dataset is small but clean.")).toBeInTheDocument();
    expect(screen.getByText("Overview")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /hist of age/i })).toBeInTheDocument();
    expect(api.generateReport).toHaveBeenCalledWith(RUN_KEY, expect.objectContaining({ signal: expect.anything() }));
  });

  it("still shows findings when the narrative could not be written", async () => {
    api.generateReport.mockResolvedValue({
      ...REPORT,
      narrative: "",
      narrative_error: "No language model is configured, so this report has no written summary.",
    });
    const user = userEvent.setup();
    render(<ReportPanel runKey={RUN_KEY} onGenerate={api.generateReport} />);

    await user.click(screen.getByRole("button", { name: /Generate report/i }));

    expect(await screen.findByText(/no written summary/i)).toBeInTheDocument();
    // Findings without narrative beats narrative without findings.
    expect(screen.getByText("240 rows and 4 columns.")).toBeInTheDocument();
  });

  it("offers the markdown source so the report can be shared", async () => {
    api.generateReport.mockResolvedValue(REPORT);
    const user = userEvent.setup();
    render(<ReportPanel runKey={RUN_KEY} onGenerate={api.generateReport} />);

    await user.click(screen.getByRole("button", { name: /Generate report/i }));

    expect(await screen.findByText(/Markdown source/i)).toBeInTheDocument();
    expect(screen.getByText(/# Analysis report/)).toBeInTheDocument();
  });

  it("surfaces a failure instead of a blank panel", async () => {
    api.generateReport.mockRejectedValue(new Error("No data snapshot for this run."));
    const user = userEvent.setup();
    render(<ReportPanel runKey={RUN_KEY} onGenerate={api.generateReport} />);

    await user.click(screen.getByRole("button", { name: /Generate report/i }));

    expect(await screen.findByText("No data snapshot for this run.")).toBeInTheDocument();
  });
});
