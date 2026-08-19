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
  // Phase 10. Resolved rather than bare vi.fn(): RunActions calls all three on
  // mount, and an undefined return would put the panel into its error boundary
  // instead of rendering it.
  listShareLinks: vi.fn(() => Promise.resolve({ links: [] })),
  createShareLink: vi.fn(),
  revokeShareLink: vi.fn(),
  getExportAvailability: vi.fn(() =>
    Promise.resolve({ bundle: { available: true, reason: null }, onnx: { available: false, reason: "test" } })
  ),
  downloadExport: vi.fn(),
  listReportSchedules: vi.fn(() => Promise.resolve({ schedules: [] })),
  createReportSchedule: vi.fn(),
  deleteReportSchedule: vi.fn(),
  UnauthorizedError: class UnauthorizedError extends Error {},
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
    all_model_scores: {
      RandomForest: { accuracy: 0.93, f1_macro: 0.92, balanced_accuracy: 0.91, roc_auc: 0.96 },
      DecisionTree: { accuracy: 0.81, f1_macro: 0.79, balanced_accuracy: 0.78 },
    },
    selection_metric: "f1_macro",
    cv_folds: 5,
    holdout_rows: 48,
    cv_scores: {
      RandomForest: { f1_macro: { mean: 0.9, std: 0.03, folds: [0.87, 0.9, 0.92, 0.89, 0.92] } },
      DecisionTree: { f1_macro: { mean: 0.77, std: 0.05, folds: [0.7, 0.78, 0.8, 0.76, 0.81] } },
    },
    baseline_cv: { f1_macro: { mean: 0.34, std: 0.01, folds: [0.34, 0.34, 0.34, 0.34, 0.34] } },
    per_class: [
      { label: "no", precision: 0.94, recall: 0.95, f1: 0.945, support: 30 },
      { label: "yes", precision: 0.88, recall: 0.86, f1: 0.87, support: 18 },
    ],
    imbalance: {
      imbalanced: false,
      severe: false,
      ratio: 1.67,
      warnings: [],
      distribution: [
        { label: "no", count: 150, share: 0.625 },
        { label: "yes", count: 90, share: 0.375 },
      ],
    },
    class_weighting: {},
    smote_used: false,
    tuning: [],
    tuning_budget_seconds: 0,
    ensemble_members: [],
    model_card: {
      schema: 1,
      environment: { libraries: { "scikit-learn": "1.6.1", xgboost: "3.4.1", catboost: null } },
      data: { target: "churn", features: { numeric: ["age", "spend"], one_hot: ["region"], target_encoded: [] } },
      training: { selection_metric: "f1_macro", candidates: ["RandomForest", "DecisionTree"], cv_folds: 5 },
      metrics: {},
    },
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
      metric: "f1_macro",
      metric_label: "macro F1",
      score: 0.92,
      baseline: 0.34,
      cv_mean: 0.9,
      cv_std: 0.03,
      holdout_rows: 48,
      reasons: ["Beats the naive baseline by 0.58 on macro F1."],
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
          all_model_scores: { DecisionTree: { accuracy: 0.017, f1_macro: 0.012 } },
          cv_scores: { DecisionTree: { f1_macro: { mean: 0.011, std: 0.004, folds: [0.01, 0.012] } } },
          health: {
            verdict: "unreliable",
            headline: "This model is not better than guessing",
            metric: "f1_macro",
            metric_label: "macro F1",
            score: 0.012,
            baseline: 0.02,
            reasons: ["Macro F1 of 0.012 does not beat the 0.020 baseline."],
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

  /** The scores chart, addressed by its heading rather than by loose text. */
  function scoresFrame() {
    return screen.getByRole("heading", { name: /Cross-validated macro F1/i }).closest("section");
  }

  it("shows every model's score as a cross-validated mean, not a single split", async () => {
    await renderDashboard();
    const frame = scoresFrame();
    expect(frame).toBeInTheDocument();
    expect(within(frame).getByText(/Mean \+\/- 1 sd over 5 folds/i)).toBeInTheDocument();
  });

  it("puts the spread next to every model in the table view", async () => {
    const user = userEvent.setup();
    await renderDashboard();
    const frame = scoresFrame();

    await user.click(within(frame).getByRole("button", { name: /Table/i }));

    // The winner is named as the winner, and its variance is shown, not hidden.
    expect(within(frame).getByText("RandomForest (selected)")).toBeInTheDocument();
    expect(within(frame).getByText("+/- 0.0300")).toBeInTheDocument();
    expect(within(frame).getByText("DecisionTree")).toBeInTheDocument();
    expect(within(frame).getByText("+/- 0.0500")).toBeInTheDocument();
  });

  it("leads the metric tiles with macro F1 rather than accuracy", async () => {
    await renderDashboard();
    const tiles = Array.from(document.querySelectorAll(".stat-label")).map((node) => node.textContent);
    // Accuracy is still reported; it is just no longer the headline number.
    expect(tiles.indexOf("macro F1")).toBeGreaterThanOrEqual(0);
    expect(tiles.indexOf("macro F1")).toBeLessThan(tiles.indexOf("accuracy"));
    expect(tiles).toContain("balanced accuracy");
    expect(tiles).toContain("ROC-AUC");
  });

  it("names the metric the verdict was actually made on", async () => {
    await renderDashboard();
    const banner = screen.getByText(/Model beats the naive baseline/i).closest(".card");
    expect(banner.textContent).toMatch(/Macro F1 0\.9200 vs naive baseline 0\.3400/);
    expect(banner.textContent).toMatch(/always predicting the most common class/);
  });
});

// --- model card ---------------------------------------------------------

describe("model card", () => {
  beforeEach(stubApi);

  async function renderCard() {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result: completedResult() }]);
    render(<UploadPage />);
    await chooseFile(user);
    await trainWithTarget(user);
    await screen.findByText(/Model card/i);
    return user;
  }

  function cardRoot() {
    return screen.getByRole("heading", { name: /Model card/i }).closest(".card");
  }

  it("states that selection was cross-validated and on which metric", async () => {
    await renderCard();
    const card = cardRoot();
    expect(within(card).getByText(/Selected on cross-validated/i)).toBeInTheDocument();
    expect(within(card).getByText("0.9000 ± 0.0300")).toBeInTheDocument();
    expect(within(card).getByText(/over 5 folds/i)).toBeInTheDocument();
    expect(within(card).getByText(/on 48 unseen rows/i)).toBeInTheDocument();
  });

  it("breaks performance down per class, so a failing class cannot hide", async () => {
    await renderCard();
    const table = screen.getByRole("table", { name: /Per-class performance/i });
    const row = within(table).getByRole("row", { name: /^yes/ });

    expect(within(row).getByText("0.880")).toBeInTheDocument(); // precision
    expect(within(row).getByText("0.860")).toBeInTheDocument(); // recall
    expect(within(row).getByText("18")).toBeInTheDocument(); // support
  });

  it("shows the class balance the run was trained against", async () => {
    await renderCard();
    const card = cardRoot();
    expect(within(card).getByText("150 (62.5%)")).toBeInTheDocument();
    expect(within(card).getByText("90 (37.5%)")).toBeInTheDocument();
  });

  it("records the library versions that produced the score", async () => {
    const user = await renderCard();
    const card = cardRoot();
    await user.click(within(card).getByRole("button", { name: /Library versions/i }));

    expect(await within(card).findByText("1.6.1")).toBeInTheDocument();
    // An absent library explains an absent model, so record it as absent.
    expect(within(card).getByText("not installed")).toBeInTheDocument();
  });
});

describe("imbalanced runs", () => {
  beforeEach(stubApi);

  it("warns about class imbalance instead of showing a flattering accuracy", async () => {
    const user = userEvent.setup();
    pollSequence([
      {
        state: "completed",
        progress: 100,
        result: completedResult({
          all_model_scores: { RandomForest: { accuracy: 0.95, f1_macro: 0.49, balanced_accuracy: 0.5 } },
          imbalance: {
            imbalanced: true,
            severe: true,
            ratio: 19,
            warnings: ["Class imbalance: 'no' is 95.0% of rows (19.0:1 against the rarest class)."],
            distribution: [
              { label: "no", count: 190, share: 0.95 },
              { label: "yes", count: 10, share: 0.05 },
            ],
          },
          class_weighting: { RandomForest: "class_weight=balanced" },
          health: {
            verdict: "unreliable",
            headline: "This model is not better than guessing",
            metric: "f1_macro",
            metric_label: "macro F1",
            score: 0.49,
            baseline: 0.487,
            reasons: [
              "Accuracy looks high at 95.0%, but that is what a constant predictor scores on this target.",
            ],
          },
        }),
      },
    ]);
    render(<UploadPage />);
    await chooseFile(user);
    await trainWithTarget(user);

    expect(await screen.findByText(/Class imbalance: 'no' is 95.0% of rows/i)).toBeInTheDocument();
    expect(screen.getByText(/what a constant predictor scores/i)).toBeInTheDocument();
    expect(screen.getByText(/not better than guessing/i)).toBeInTheDocument();
    // Named per strategy, with the model it applied to, not repeated per model.
    expect(screen.getByText("class_weight=balanced (RandomForest)")).toBeInTheDocument();
  });
});

// --- training options ---------------------------------------------------

describe("training options", () => {
  beforeEach(stubApi);

  it("sends a tuning budget and the SMOTE choice with the job", async () => {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result: completedResult() }]);
    render(<UploadPage />);

    await chooseFile(user);
    await user.selectOptions(screen.getByLabelText(/Hyperparameter search/i), "120");
    await user.click(screen.getByRole("checkbox", { name: /Oversample the minority class/i }));
    await trainWithTarget(user);

    await waitFor(() => expect(api.startUploadJob).toHaveBeenCalled());
    const formData = api.startUploadJob.mock.calls[0][0];
    expect(formData.get("tuning_budget_seconds")).toBe("120");
    expect(formData.get("use_smote")).toBe("true");
  });

  it("defaults to library defaults and no oversampling", async () => {
    const user = userEvent.setup();
    pollSequence([{ state: "completed", progress: 100, result: completedResult() }]);
    render(<UploadPage />);

    await chooseFile(user);
    await trainWithTarget(user);

    await waitFor(() => expect(api.startUploadJob).toHaveBeenCalled());
    const formData = api.startUploadJob.mock.calls[0][0];
    expect(formData.get("tuning_budget_seconds")).toBe("0");
    expect(formData.get("use_smote")).toBe("false");
  });
});
