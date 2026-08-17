import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

// ESM bindings are live, so spying on the namespace object does not replace
// what UploadPage already imported. Mock the module itself.
vi.mock("../services/api", () => ({
  startUploadJob: vi.fn(),
  getUploadJobStatus: vi.fn(),
  predictDataset: vi.fn(),
  chatQuery: vi.fn(),
}));

import UploadPage from "../pages/UploadPage";
import * as api from "../services/api";

const CSV = "age,region,spend,churn\n30,north,50,yes\n60,south,40,no\n";

function csvFile(name = "data.csv") {
  return new File([CSV], name, { type: "text/csv" });
}

function completedResult(overrides = {}) {
  return {
    status: "trained",
    run_key: "a".repeat(64),
    problem_type: "classification",
    mode: "auto",
    selected_model: "RandomForest",
    target: "churn",
    features: ["age", "region", "spend"],
    all_model_scores: { RandomForest: { accuracy: 0.93 } },
    feature_importance: [
      { feature: "region", importance: 0.6 },
      { feature: "age", importance: 0.3 },
    ],
    explanation_method: "shap",
    failed_models: {},
    insights: ["Target column: 'churn' (auto-detected)."],
    health: {
      verdict: "strong",
      headline: "Model beats the naive baseline",
      metric: "accuracy",
      score: 0.93,
      baseline: 0.5,
      reasons: ["Beats the naive baseline by 0.43 on accuracy."],
    },
    ...overrides,
  };
}

/** Queue of statuses returned by successive polls. */
function pollSequence(statuses) {
  let call = 0;
  api.getUploadJobStatus.mockImplementation(async () => {
    const status = statuses[Math.min(call, statuses.length - 1)];
    call += 1;
    return status;
  });
}

async function submitUpload(user, { targetColumn } = {}) {
  const fileInput = document.querySelector('input[type="file"]');
  await user.upload(fileInput, csvFile());

  if (targetColumn) {
    // The picker only populates once the header has been parsed.
    await waitFor(() => expect(screen.getByRole("option", { name: targetColumn })).toBeInTheDocument());
    await user.selectOptions(screen.getByLabelText(/Target Column/i), targetColumn);
  }

  // The submit button is real and enabled, but jsdom's constraint validation
  // does not treat an assigned FileList as satisfying `required`, so a click
  // is swallowed here (it works in a browser). Dispatch submit directly —
  // what these tests exercise is everything after submission.
  expect(screen.getByRole("button", { name: /Start Autonomous Pipeline/i })).toBeEnabled();
  fireEvent.submit(document.querySelector("form"));
}

describe("upload -> poll -> render", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.startUploadJob.mockResolvedValue({ job_id: "job-1", state: "queued" });
  });

  it("shows progress while running, then renders the dashboard", async () => {
    const user = userEvent.setup();
    pollSequence([
      { state: "running", progress: 60, current_step: "Training models", status_log: [{ step: "Training models" }] },
      { state: "completed", progress: 100, result: completedResult() },
    ]);

    render(<UploadPage />);
    await submitUpload(user);

    // Appears twice: as the current step and in the status log.
    expect((await screen.findAllByText(/Training models/i)).length).toBeGreaterThan(0);
    expect(await screen.findByText(/Model beats the naive baseline/i)).toBeInTheDocument();
    expect(screen.getByText("RandomForest")).toBeInTheDocument();
    expect(screen.getByText(/Selected Model$/i)).toBeInTheDocument();
  });

  it("sends the chosen target column to the API", async () => {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result: completedResult({ target: "region" }) }]);

    render(<UploadPage />);
    await submitUpload(user, { targetColumn: "region" });

    await waitFor(() => expect(api.startUploadJob).toHaveBeenCalled());
    const formData = api.startUploadJob.mock.calls[0][0];
    expect(formData.get("target_column")).toBe("region");
    expect(formData.get("mode")).toBe("auto");
  });

  it("omits target_column when auto-detect is left selected", async () => {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result: completedResult() }]);

    render(<UploadPage />);
    await submitUpload(user);

    await waitFor(() => expect(api.startUploadJob).toHaveBeenCalled());
    expect(api.startUploadJob.mock.calls[0][0].get("target_column")).toBeNull();
  });

  it("labels a rejected target as a target problem, not a crash", async () => {
    const user = userEvent.setup();
    pollSequence([
      {
        state: "failed",
        error: "'Findings' has 14,555 unique values across 19,245 rows — this looks like free text, not a label.",
        error_kind: "target",
      },
    ]);

    render(<UploadPage />);
    await submitUpload(user);

    expect(await screen.findByText(/Unusable Target Column/i)).toBeInTheDocument();
    expect(screen.getByText(/looks like free text, not a label/i)).toBeInTheDocument();
    expect(screen.getByText(/Pick a different column/i)).toBeInTheDocument();
  });

  it("surfaces a low-confidence verdict instead of calling it optimized", async () => {
    const user = userEvent.setup();
    pollSequence([
      {
        state: "completed",
        progress: 100,
        result: completedResult({
          selected_model: "DecisionTree",
          all_model_scores: { DecisionTree: { accuracy: 0.017 } },
          health: {
            verdict: "unreliable",
            headline: "This model is not better than guessing",
            metric: "accuracy",
            score: 0.017,
            baseline: 0.02,
            reasons: ["Accuracy of 1.7% does not beat the 2.0% you get from always predicting the most common class."],
          },
        }),
      },
    ]);

    render(<UploadPage />);
    await submitUpload(user);

    expect(await screen.findByText(/not better than guessing/i)).toBeInTheDocument();
    expect(screen.getByText(/Selected Model \(low confidence\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/^Optimized Model$/i)).not.toBeInTheDocument();
  });

  it("lists models that failed to train", async () => {
    const user = userEvent.setup();
    pollSequence([
      {
        state: "completed",
        progress: 100,
        result: completedResult({ failed_models: { XGBoost: "Invalid classes inferred from unique values of `y`" } }),
      },
    ]);

    render(<UploadPage />);
    await submitUpload(user);

    expect(await screen.findByText(/Models That Failed \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Invalid classes inferred/i)).toBeInTheDocument();
  });

  it("names the attribution method next to the importances", async () => {
    const user = userEvent.setup();
    pollSequence([
      { state: "completed", progress: 100, result: completedResult({ explanation_method: "native" }) },
    ]);

    render(<UploadPage />);
    await submitUpload(user);

    expect(await screen.findByText(/model-native importances \(SHAP unavailable\)/i)).toBeInTheDocument();
  });
});
