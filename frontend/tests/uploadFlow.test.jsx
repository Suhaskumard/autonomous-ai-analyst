import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

// ESM bindings are live, so spying on the namespace object does not replace
// what the page already imported. Mock the module itself.
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
}));

import UploadPage from "../pages/UploadPage";
import * as api from "../services/api";

const CSV = "age,region,spend,churn\n30,north,50,yes\n60,south,40,no\n";

function csvFile(name = "data.csv") {
  return new File([CSV], name, { type: "text/csv" });
}

function col(overrides = {}) {
  return {
    name: "col",
    dtype: "categorical",
    missing: 0,
    missing_pct: 0,
    unique: 5,
    unique_pct: 2,
    usable_as_target: true,
    target_reason: "5 classes — usable as a classification target.",
    feature_note: null,
    sparkline: { kind: "categories", labels: ["a", "b"], counts: [3, 2] },
    ...overrides,
  };
}

function profilePayload(overrides = {}) {
  return {
    filename: "data.csv",
    rows: 240,
    columns: 4,
    suggested_target: "churn",
    parse_report: { rows_parsed: 240, rows_skipped: 0, delimiter: ",", encoding: "utf-8", warnings: [] },
    column_profiles: [
      col({ name: "age", dtype: "numeric" }),
      col({ name: "region", unique: 3 }),
      col({ name: "spend", dtype: "numeric" }),
      col({ name: "churn", unique: 2 }),
    ],
    ...overrides,
  };
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
    all_model_scores: { RandomForest: { accuracy: 0.93 }, DecisionTree: { accuracy: 0.81 } },
    feature_importance: [
      { feature: "region", importance: 0.6 },
      { feature: "age", importance: 0.3 },
    ],
    explanation_method: "shap",
    failed_models: {},
    insights: ["Target column: 'churn' (auto-detected)."],
    parse_report: { rows_parsed: 240, rows_skipped: 0, delimiter: ",", encoding: "utf-8", warnings: [] },
    quality_report: { warnings: [] },
    charts: { numeric_histograms: {}, categorical_bars: {} },
    diagnostics: {
      problem_type: "classification",
      confusion_matrix: {
        labels: ["no", "yes"],
        matrix: [
          [30, 4],
          [3, 25],
        ],
        total: 62,
        truncated: false,
      },
      residuals: null,
      correlation: {
        columns: ["age", "spend"],
        matrix: [
          [1, 0.12],
          [0.12, 1],
        ],
        truncated: false,
        strongest_pairs: [{ a: "age", b: "spend", r: 0.12 }],
      },
    },
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

function pollSequence(statuses) {
  let call = 0;
  api.getUploadJobStatus.mockImplementation(async () => {
    const status = statuses[Math.min(call, statuses.length - 1)];
    call += 1;
    return status;
  });
}

/** Choose a file and wait for the profile screen to appear. */
async function chooseFile(user) {
  const input = document.querySelector('input[type="file"]');
  await user.upload(input, csvFile());
  await screen.findByText(/usable as a target/i);
}

async function trainWithTarget(user, target = "churn") {
  await user.click(screen.getByRole("radio", { name: new RegExp(`Predict ${target}`, "i") }));
  await user.click(screen.getByRole("button", { name: /Train on this dataset/i }));
}

function stubApi() {
  vi.clearAllMocks();
  api.startUploadJob.mockResolvedValue({ job_id: "job-1", state: "queued" });
  api.profileDataset.mockResolvedValue(profilePayload());
  api.listRuns.mockResolvedValue({ count: 0, runs: [] });
}

describe("profile -> train -> dashboard", () => {
  beforeEach(stubApi);

  it("profiles the file before anything is trained", async () => {
    const user = userEvent.setup();
    render(<UploadPage />);

    await chooseFile(user);

    expect(api.profileDataset).toHaveBeenCalledTimes(1);
    expect(api.startUploadJob).not.toHaveBeenCalled();
    expect(screen.getByText("240 rows · 4 columns · 4 usable as a target")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /Predict churn/i })).toBeInTheDocument();
  });

  it("sends the chosen target column when training starts", async () => {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result: completedResult({ target: "region" }) }]);
    render(<UploadPage />);

    await chooseFile(user);
    await trainWithTarget(user, "region");

    await waitFor(() => expect(api.startUploadJob).toHaveBeenCalled());
    const formData = api.startUploadJob.mock.calls[0][0];
    expect(formData.get("target_column")).toBe("region");
    expect(formData.get("mode")).toBe("auto");
  });

  it("disables columns that cannot be a target and says why", async () => {
    const user = userEvent.setup();
    api.profileDataset.mockResolvedValue(
      profilePayload({
        column_profiles: [
          col({ name: "age", dtype: "numeric" }),
          col({
            name: "Findings",
            usable_as_target: false,
            target_reason: "14,555 unique values across 19,245 rows — looks like free text or an ID.",
            feature_note: "Will be dropped as free text / identifier.",
          }),
        ],
      })
    );
    render(<UploadPage />);
    await chooseFile(user);

    expect(screen.getByRole("radio", { name: /Predict Findings/i })).toBeDisabled();
    expect(screen.getByText(/looks like free text or an ID/i)).toBeInTheDocument();
    expect(screen.getByText(/Will be dropped as free text/i)).toBeInTheDocument();
  });

  it("shows progress while running, then the dashboard", async () => {
    const user = userEvent.setup();
    pollSequence([
      { state: "running", progress: 60, current_step: "Training models", status_log: [{ step: "Training models" }] },
      { state: "completed", progress: 100, result: completedResult() },
    ]);
    render(<UploadPage />);

    await chooseFile(user);
    await trainWithTarget(user);

    expect((await screen.findAllByText(/Training models/i)).length).toBeGreaterThan(0);
    expect(await screen.findByText(/Model beats the naive baseline/i)).toBeInTheDocument();
    expect(screen.getByText("RandomForest")).toBeInTheDocument();
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

    await chooseFile(user);
    await trainWithTarget(user);

    expect(await screen.findByText(/Unusable target column/i)).toBeInTheDocument();
    expect(screen.getByText(/looks like free text, not a label/i)).toBeInTheDocument();
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
            reasons: ["Accuracy of 1.7% does not beat the 2.0% baseline."],
          },
        }),
      },
    ]);
    render(<UploadPage />);

    await chooseFile(user);
    await trainWithTarget(user);

    expect(await screen.findByText(/not better than guessing/i)).toBeInTheDocument();
    expect(screen.getByText(/Selected model \(low confidence\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/^Optimized Model$/i)).not.toBeInTheDocument();
  });

  it("says when a result was replayed from cache rather than trained", async () => {
    const user = userEvent.setup();
    pollSequence([
      {
        state: "completed",
        progress: 100,
        result: completedResult({
          status: "reused",
          message: "This dataset, mode, and target combination was already trained. Reused stored artifacts.",
        }),
      },
    ]);
    render(<UploadPage />);

    await chooseFile(user);
    await trainWithTarget(user);

    // A replay and a fresh run used to look identical.
    expect(await screen.findByText(/already trained\. Reused stored artifacts/i)).toBeInTheDocument();
  });

  it("marks a freshly trained run as fresh", async () => {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result: completedResult() }]);
    render(<UploadPage />);

    await chooseFile(user);
    await trainWithTarget(user);

    expect(await screen.findByText(/Freshly trained just now/i)).toBeInTheDocument();
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

    await chooseFile(user);
    await trainWithTarget(user);

    expect(await screen.findByText(/Models that failed \(1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Invalid classes inferred/i)).toBeInTheDocument();
  });

  it("reports how the file was read, including dropped values", async () => {
    const user = userEvent.setup();
    pollSequence([
      {
        state: "completed",
        progress: 100,
        result: completedResult({
          parse_report: {
            rows_parsed: 238,
            rows_skipped: 0,
            rows_truncated: 2,
            delimiter: ";",
            encoding: "utf-8",
            warnings: ["2 row(s) had more values than the header has columns; the extra values were dropped."],
          },
        }),
      },
    ]);
    render(<UploadPage />);

    await chooseFile(user);
    await trainWithTarget(user);

    expect(await screen.findByText(/How your file was read/i)).toBeInTheDocument();
    expect(screen.getByText(/2 row\(s\) had more values than the header/i)).toBeInTheDocument();
  });
});

describe("charts", () => {
  beforeEach(stubApi);

  async function renderDashboard(result = completedResult()) {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result }]);
    render(<UploadPage />);
    await chooseFile(user);
    await trainWithTarget(user);
    await screen.findByText(/Key feature influencers/i);
    return user;
  }

  it("renders the confusion matrix with real class names", async () => {
    await renderDashboard();
    expect(screen.getByText(/Confusion matrix/i)).toBeInTheDocument();
    expect(screen.getByTitle("True no, predicted no: 30")).toBeInTheDocument();
    expect(screen.getByTitle("True yes, predicted no: 3")).toBeInTheDocument();
  });

  it("renders a residual plot for regression instead of a matrix", async () => {
    await renderDashboard(
      completedResult({
        problem_type: "regression",
        target: "price",
        selected_model: "LinearRegression",
        all_model_scores: { LinearRegression: { rmse: 12.5 } },
        health: {
          verdict: "strong",
          headline: "Model beats the naive baseline",
          metric: "rmse",
          score: 12.5,
          baseline: 90,
          reasons: [],
        },
        diagnostics: {
          problem_type: "regression",
          confusion_matrix: null,
          correlation: null,
          residuals: {
            points: [{ predicted: 10, residual: 1, actual: 11 }],
            sampled: false,
            total: 1,
            mean_residual: 1,
            std_residual: 0,
          },
        },
      })
    );
    expect(screen.getByText(/Residuals vs predicted/i)).toBeInTheDocument();
    expect(screen.queryByText(/Confusion matrix/i)).not.toBeInTheDocument();
  });

  it("names the attribution method next to the importances", async () => {
    await renderDashboard(completedResult({ explanation_method: "native" }));
    expect(screen.getByText(/model-native importances \(SHAP unavailable\)/i)).toBeInTheDocument();
  });

  it("offers a table view of every chart as the accessible fallback", async () => {
    const user = await renderDashboard();
    const frame = screen.getByText(/Key feature influencers/i).closest("section");
    await user.click(within(frame).getByRole("button", { name: /Table/i }));
    expect(within(frame).getByRole("table")).toBeInTheDocument();
  });

  it("explains rather than plots when a correlation is unavailable", async () => {
    await renderDashboard(
      completedResult({
        diagnostics: { problem_type: "classification", confusion_matrix: null, residuals: null, correlation: null },
      })
    );
    expect(screen.getByText(/at least two varying numeric columns/i)).toBeInTheDocument();
  });

  it("shows every model's score, not only the winner's", async () => {
    await renderDashboard();
    const frame = screen.getByText(/Model scores \(accuracy\)/i).closest("section");
    expect(frame).toBeInTheDocument();
  });
});
