import React from "react";
import { render, renderHook, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { act } from "react";
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
}));

import Workspace from "../components/Workspace";
import useUploadJob, { POLL_MAX_MS, nextDelay } from "../hooks/useUploadJob";
import * as api from "../services/api";

// --- polling -----------------------------------------------------------

describe("useUploadJob polling", () => {
  beforeEach(() => vi.clearAllMocks());

  it("backs off between polls and stops climbing at the ceiling", () => {
    let delay = 700;
    const seen = [delay];
    for (let i = 0; i < 12; i += 1) {
      delay = nextDelay(delay);
      seen.push(delay);
    }
    // Strictly increasing until it saturates — never a fixed 1s hammer.
    expect(seen[1]).toBeGreaterThan(seen[0]);
    expect(Math.max(...seen)).toBe(POLL_MAX_MS);
    expect(seen[seen.length - 1]).toBe(POLL_MAX_MS);
  });

  it("stops polling when the component unmounts", async () => {
    api.startUploadJob.mockResolvedValue({ job_id: "job-1" });
    api.getUploadJobStatus.mockResolvedValue({ state: "running", progress: 10, status_log: [] });

    const { result, unmount } = renderHook(() => useUploadJob());
    await act(async () => {
      result.current.run(new FormData());
    });
    await waitFor(() => expect(api.getUploadJobStatus).toHaveBeenCalled());

    const callsAtUnmount = api.getUploadJobStatus.mock.calls.length;
    unmount();
    await new Promise((resolve) => setTimeout(resolve, 900));

    // The abort on unmount stops the loop; a stray extra in-flight call may
    // land, but it must not keep polling.
    expect(api.getUploadJobStatus.mock.calls.length).toBeLessThanOrEqual(callsAtUnmount + 1);
  });

  it("passes an abort signal to every request", async () => {
    api.startUploadJob.mockResolvedValue({ job_id: "job-1" });
    api.getUploadJobStatus.mockResolvedValue({ state: "completed", result: { run_key: "a".repeat(64) } });

    const { result } = renderHook(() => useUploadJob());
    await act(async () => {
      await result.current.run(new FormData());
    });

    expect(api.startUploadJob.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal);
    expect(api.getUploadJobStatus.mock.calls[0][1].signal).toBeInstanceOf(AbortSignal);
  });

  it("surfaces a failed job with its kind", async () => {
    api.startUploadJob.mockResolvedValue({ job_id: "job-1" });
    api.getUploadJobStatus.mockResolvedValue({ state: "failed", error: "bad target", error_kind: "target" });

    const { result } = renderHook(() => useUploadJob());
    await act(async () => {
      await result.current.run(new FormData());
    });

    expect(result.current.error).toBe("bad target");
    expect(result.current.errorKind).toBe("target");
    expect(result.current.loading).toBe(false);
  });

  it("reports a network failure instead of hanging", async () => {
    api.startUploadJob.mockRejectedValue(new Error("Failed to fetch"));

    const { result } = renderHook(() => useUploadJob());
    await act(async () => {
      await result.current.run(new FormData());
    });

    expect(result.current.errorKind).toBe("network");
    expect(result.current.loading).toBe(false);
  });

  it("swallows a deliberate cancel rather than showing an error", async () => {
    api.startUploadJob.mockResolvedValue({ job_id: "job-1" });
    api.getUploadJobStatus.mockResolvedValue({ state: "running", progress: 5, status_log: [] });

    const { result } = renderHook(() => useUploadJob());
    await act(async () => {
      result.current.run(new FormData());
    });
    await waitFor(() => expect(api.getUploadJobStatus).toHaveBeenCalled());
    act(() => result.current.cancel());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe("");
  });
});

// --- workspace ---------------------------------------------------------

const RUN = {
  run_key: "b".repeat(64),
  filename: "customers.csv",
  target: "churn",
  mode: "auto",
  selected_model: "RandomForest",
  health_verdict: "strong",
  row_count: 240,
  column_count: 4,
  created_at: new Date(Date.now() - 3600_000).toISOString(),
};

describe("Workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listRuns.mockResolvedValue({ count: 1, runs: [RUN] });
  });

  it("lists past runs so yesterday's dataset can be found", async () => {
    render(<Workspace onOpenRun={vi.fn()} />);

    expect(await screen.findByText("customers.csv")).toBeInTheDocument();
    expect(screen.getByText("churn")).toBeInTheDocument();
    expect(screen.getByText("1h ago")).toBeInTheDocument();
    // The verdict is a word, not only a colour.
    expect(screen.getByText("Healthy")).toBeInTheDocument();
  });

  it("reopens a run into the dashboard", async () => {
    const onOpenRun = vi.fn();
    const reopened = { run_key: RUN.run_key, status: "reopened" };
    api.getRunResult.mockResolvedValue(reopened);
    const user = userEvent.setup();

    render(<Workspace onOpenRun={onOpenRun} />);
    await user.click(await screen.findByRole("button", { name: /Open/i }));

    await waitFor(() => expect(api.getRunResult).toHaveBeenCalledWith(RUN.run_key));
    expect(onOpenRun).toHaveBeenCalledWith(reopened);
  });

  it("deletes a run and refreshes the list", async () => {
    api.deleteRun.mockResolvedValue({ deleted: true });
    api.listRuns.mockResolvedValueOnce({ count: 1, runs: [RUN] }).mockResolvedValueOnce({ count: 0, runs: [] });
    const user = userEvent.setup();

    render(<Workspace onOpenRun={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: /Delete customers.csv/i }));

    await waitFor(() => expect(api.deleteRun).toHaveBeenCalledWith(RUN.run_key));
    expect(await screen.findByText(/No past runs yet/i)).toBeInTheDocument();
  });

  it("compares selected runs", async () => {
    const second = { ...RUN, run_key: "c".repeat(64), filename: "customers-ensemble.csv", mode: "ensemble" };
    api.listRuns.mockResolvedValue({ count: 2, runs: [RUN, second] });
    api.compareRuns.mockResolvedValue({
      comparable: true,
      metric: "accuracy",
      problem_type: "classification",
      runs: [
        { run_key: RUN.run_key, mode: "auto", target: "churn", scores: { RandomForest: { accuracy: 0.9 } } },
        { run_key: second.run_key, mode: "ensemble", target: "churn", scores: { RandomForest: { accuracy: 0.93 } } },
      ],
    });
    const user = userEvent.setup();

    render(<Workspace onOpenRun={vi.fn()} />);
    await user.click(await screen.findByRole("checkbox", { name: /Compare customers.csv/i }));
    await user.click(screen.getByRole("checkbox", { name: /Compare customers-ensemble.csv/i }));
    await user.click(screen.getByRole("button", { name: /Compare 2/i }));

    await waitFor(() => expect(api.compareRuns).toHaveBeenCalledWith([RUN.run_key, second.run_key]));
    expect(await screen.findByText(/Compare runs \(accuracy\)/i)).toBeInTheDocument();
  });

  it("refuses to plot runs of different problem types", async () => {
    const second = { ...RUN, run_key: "c".repeat(64), filename: "houses.csv" };
    api.listRuns.mockResolvedValue({ count: 2, runs: [RUN, second] });
    api.compareRuns.mockResolvedValue({ comparable: false, metric: null, runs: [] });
    const user = userEvent.setup();

    render(<Workspace onOpenRun={vi.fn()} />);
    await user.click(await screen.findByRole("checkbox", { name: /Compare customers.csv/i }));
    await user.click(screen.getByRole("checkbox", { name: /Compare houses.csv/i }));
    await user.click(screen.getByRole("button", { name: /Compare 2/i }));

    expect(await screen.findByText(/cannot share an axis/i)).toBeInTheDocument();
  });

  it("says so plainly when there is nothing stored yet", async () => {
    api.listRuns.mockResolvedValue({ count: 0, runs: [] });
    render(<Workspace onOpenRun={vi.fn()} />);
    expect(await screen.findByText(/No past runs yet/i)).toBeInTheDocument();
  });
});

// --- run preview: the caller /api/insights never had ---------------------

const INSIGHTS = {
  run_key: RUN.run_key,
  model: "RandomForest",
  target: "churn",
  problem_type: "classification",
  scores: { RandomForest: { accuracy: 0.93, f1_score: 0.9 } },
  health: { verdict: "strong", headline: "Model beats the naive baseline" },
  failed_models: { XGBoost: "Invalid classes inferred from unique values of `y`." },
  feature_importance: [
    { feature: "region", importance: 0.61 },
    { feature: "age", importance: 0.28 },
  ],
  explanation_method: "shap",
  insights: [],
  quality_report: {},
  charts: {},
  summary_stats: {},
};

describe("Workspace run preview", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listRuns.mockResolvedValue({ count: 1, runs: [RUN] });
    api.getInsights.mockResolvedValue(INSIGHTS);
  });

  it("summarises a past run without reopening it", async () => {
    const onOpenRun = vi.fn();
    const user = userEvent.setup();

    render(<Workspace onOpenRun={onOpenRun} />);
    await user.click(await screen.findByRole("button", { name: /Summary of customers.csv/i }));

    await waitFor(() => expect(api.getInsights).toHaveBeenCalledWith(RUN.run_key));
    expect(await screen.findByText("Model beats the naive baseline")).toBeInTheDocument();
    expect(screen.getByText("region")).toBeInTheDocument();
    expect(screen.getByText("accuracy")).toBeInTheDocument();
    expect(screen.getByText(/Failed to train: XGBoost/)).toBeInTheDocument();
    // The point of the cheap read: the full dashboard payload is never fetched.
    expect(api.getRunResult).not.toHaveBeenCalled();
    expect(onOpenRun).not.toHaveBeenCalled();
  });

  it("collapses again and does not refetch a summary it already has", async () => {
    const user = userEvent.setup();
    render(<Workspace onOpenRun={vi.fn()} />);
    const toggle = await screen.findByRole("button", { name: /Summary of customers.csv/i });

    await user.click(toggle);
    expect(await screen.findByText("Model beats the naive baseline")).toBeInTheDocument();

    await user.click(toggle);
    await waitFor(() => expect(screen.queryByText("Model beats the naive baseline")).not.toBeInTheDocument());

    await user.click(toggle);
    expect(await screen.findByText("Model beats the naive baseline")).toBeInTheDocument();
    expect(api.getInsights).toHaveBeenCalledTimes(1);
  });

  it("reports a summary that could not be loaded", async () => {
    api.getInsights.mockRejectedValue(new Error("No metadata found for this run key."));
    const user = userEvent.setup();

    render(<Workspace onOpenRun={vi.fn()} />);
    await user.click(await screen.findByRole("button", { name: /Summary of customers.csv/i }));

    expect(await screen.findByText(/No metadata found for this run key/i)).toBeInTheDocument();
    // The row list itself is still usable.
    expect(screen.getByText("customers.csv")).toBeInTheDocument();
  });
});
