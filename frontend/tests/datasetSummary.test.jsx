import React from "react";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import DatasetSummary from "../components/DatasetSummary";

/**
 * Phase 3's exit criterion is that every field the backend returns is visible
 * somewhere. summary_stats, dropped_columns, class_names, the trained shape
 * and the counted half of the quality report had no home in the UI; this is
 * the guard that they still do.
 */

function result(overrides = {}) {
  return {
    row_count: 240,
    column_count: 4,
    mode: "ensemble",
    dataset_hash: "d".repeat(64),
    target: "churn",
    target_auto_detected: false,
    problem_type: "classification",
    class_names: ["no", "yes"],
    features: ["age", "region", "spend"],
    dropped_columns: ["customer_id"],
    timestamp: "2026-08-17T10:30:00+00:00",
    quality_report: { rows: 240, columns: 4, duplicate_rows: 7, missing_counts: {}, warnings: [] },
    summary_stats: {
      age: { type: "numeric", mean: 43.5, median: 42, min: 18, max: 88, std: 12.25, null_count: 0 },
      spend: { type: "numeric", mean: 61.2, median: 58, min: 0, max: 500, std: 40.1, null_count: 12 },
      region: { type: "categorical", most_frequent: "north", unique_count: 3, null_count: 0 },
    },
    ...overrides,
  };
}

function rowFor(name) {
  return screen.getByRole("row", { name: new RegExp(`^${name}\\b`) });
}

/** The value of one labelled fact, so "3 features" is not confused with "3 regions". */
function factFor(label) {
  return within(screen.getByText(label).closest(".fact"));
}

describe("DatasetSummary", () => {
  it("renders the per-column numeric statistics the backend computes", () => {
    render(<DatasetSummary result={result()} />);

    const age = rowFor("age");
    expect(within(age).getByText("43.500")).toBeInTheDocument(); // mean
    expect(within(age).getByText("42")).toBeInTheDocument(); // median
    expect(within(age).getByText("18")).toBeInTheDocument(); // min
    expect(within(age).getByText("88")).toBeInTheDocument(); // max
    expect(within(age).getByText("12.250")).toBeInTheDocument(); // std dev
  });

  it("renders categorical columns with distinct counts and the most common value", () => {
    render(<DatasetSummary result={result()} />);

    const region = rowFor("region");
    expect(within(region).getByText("3")).toBeInTheDocument();
    expect(within(region).getByText("north")).toBeInTheDocument();
  });

  it("reports missing values against the row count they came from", () => {
    render(<DatasetSummary result={result()} />);

    // A bare "12" is unreadable without its denominator.
    expect(within(rowFor("spend")).getByText("12 (5.0%)")).toBeInTheDocument();
    expect(within(rowFor("age")).getByText("none")).toBeInTheDocument();
  });

  it("shows the trained shape, classes, and duplicates removed", () => {
    render(<DatasetSummary result={result()} />);

    expect(factFor("Trained shape").getByText("240 × 4")).toBeInTheDocument();
    expect(factFor("Classes").getByText("no, yes")).toBeInTheDocument();
    expect(factFor("Duplicate rows removed").getByText("7")).toBeInTheDocument();
    expect(factFor("Features used").getByText("3")).toBeInTheDocument();
    expect(factFor("Training mode").getByText("ensemble")).toBeInTheDocument();
  });

  it("shows the dataset fingerprint that decides whether a run is replayed", () => {
    render(<DatasetSummary result={result()} />);

    const fingerprint = factFor("Dataset fingerprint").getByText(/^dddddddddddd…$/);
    // Truncated for reading, complete on hover — it is a 64-character digest.
    expect(fingerprint).toHaveAttribute("title", "d".repeat(64));
  });

  it("names the columns dropped before training and why", () => {
    render(<DatasetSummary result={result()} />);

    expect(screen.getByText("customer_id")).toBeInTheDocument();
    expect(screen.getByText(/Dropped before training/i)).toBeInTheDocument();
  });

  it("says whether the target was auto-detected or chosen", () => {
    const { unmount } = render(<DatasetSummary result={result()} />);
    expect(screen.getByText(/chosen by you/i)).toBeInTheDocument();
    unmount();

    render(<DatasetSummary result={result({ target_auto_detected: true })} />);
    expect(screen.getByText(/auto-detected/i)).toBeInTheDocument();
  });

  it("calls a regression target continuous rather than listing empty classes", () => {
    render(<DatasetSummary result={result({ problem_type: "regression", class_names: [] })} />);

    expect(screen.getByText("Predicted value")).toBeInTheDocument();
    expect(screen.getByText("continuous")).toBeInTheDocument();
  });

  it("renders nothing when the run carries no summary statistics", () => {
    const { container } = render(<DatasetSummary result={result({ summary_stats: {} })} />);
    expect(container).toBeEmptyDOMElement();
  });
});
