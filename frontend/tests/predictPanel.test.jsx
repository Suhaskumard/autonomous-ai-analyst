import React from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../services/api", () => ({
  predictDataset: vi.fn(),
  predictSingleRow: vi.fn(),
}));

import PredictPanel from "../components/PredictPanel";
import * as api from "../services/api";

const FEATURES = ["age", "region", "spend"];
const RUN_KEY = "a".repeat(64);

function response(overrides = {}) {
  return {
    run_key: RUN_KEY,
    problem_type: "classification",
    target: "churn",
    features: FEATURES,
    class_names: ["no", "yes"],
    parse_warnings: [],
    rows: [
      { index: 0, prediction: "yes", confidence: 0.91, class_probabilities: { yes: 0.91, no: 0.09 } },
      { index: 1, prediction: "no", confidence: 0.77, class_probabilities: { yes: 0.23, no: 0.77 } },
    ],
    predictions: ["yes", "no"],
    confidences: [0.91, 0.77],
    input_rows: [
      { age: 70, region: "north", spend: 51.2 },
      { age: 24, region: "south", spend: 30 },
    ],
    ...overrides,
  };
}

function renderPanel(props = {}) {
  return render(
    <PredictPanel runKey={RUN_KEY} features={FEATURES} problemType="classification" target="churn" {...props} />
  );
}

describe("PredictPanel", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders predictions as a table, not a JSON dump", async () => {
    api.predictDataset.mockResolvedValue(response());
    const user = userEvent.setup();
    renderPanel();

    await user.upload(
      screen.getByLabelText(/Inference data/i),
      new File(["age,region,spend\n70,north,51.2\n"], "p.csv", { type: "text/csv" })
    );
    expect(screen.getByRole("button", { name: /Run predictions/i })).toBeEnabled();
    fireEvent.submit(screen.getByLabelText(/Inference data/i).closest("form"));

    const table = await screen.findByRole("table");
    expect(within(table).getByText("yes")).toBeInTheDocument();
    expect(within(table).getByText("91.0%")).toBeInTheDocument();
    expect(within(table).getByText("north")).toBeInTheDocument();
    expect(screen.queryByText(/"run_key"/)).not.toBeInTheDocument();
  });

  it("predicts a single pasted row without a file", async () => {
    api.predictSingleRow.mockResolvedValue(response({ rows: [response().rows[0]], predictions: ["yes"] }));
    const user = userEvent.setup();
    renderPanel();

    await user.click(screen.getByRole("tab", { name: /Enter one row/i }));
    await user.type(screen.getByLabelText("age"), "70");
    await user.type(screen.getByLabelText("region"), "north");
    await user.type(screen.getByLabelText("spend"), "51.2");
    await user.click(screen.getByRole("button", { name: /Predict this row/i }));

    await waitFor(() =>
      expect(api.predictSingleRow).toHaveBeenCalledWith(RUN_KEY, { age: "70", region: "north", spend: "51.2" })
    );
    expect(await screen.findByRole("table")).toBeInTheDocument();
  });

  it("offers a CSV download of the predictions", async () => {
    api.predictDataset.mockResolvedValue(response());
    const createObjectURL = vi.fn(() => "blob:fake");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });
    const user = userEvent.setup();
    renderPanel();

    await user.upload(
      screen.getByLabelText(/Inference data/i),
      new File(["age,region,spend\n70,north,51.2\n"], "p.csv", { type: "text/csv" })
    );
    expect(screen.getByRole("button", { name: /Run predictions/i })).toBeEnabled();
    fireEvent.submit(screen.getByLabelText(/Inference data/i).closest("form"));
    await screen.findByRole("table");
    await user.click(screen.getByRole("button", { name: /Download CSV/i }));

    expect(createObjectURL).toHaveBeenCalled();
    const blob = createObjectURL.mock.calls[0][0];
    const text = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(blob);
    });
    expect(text.split("\n")[0]).toBe("age,region,spend,predicted_churn,confidence");
    expect(text).toContain("70,north,51.2,yes,0.9100");
    vi.unstubAllGlobals();
  });

  it("says plainly when a model has no confidence to report", async () => {
    api.predictDataset.mockResolvedValue(
      response({
        confidences: null,
        rows: [{ index: 0, prediction: "yes" }],
        predictions: ["yes"],
        input_rows: [{ age: 70, region: "north", spend: 51.2 }],
      })
    );
    const user = userEvent.setup();
    renderPanel();

    await user.upload(
      screen.getByLabelText(/Inference data/i),
      new File(["age,region,spend\n70,north,51.2\n"], "p.csv", { type: "text/csv" })
    );
    expect(screen.getByRole("button", { name: /Run predictions/i })).toBeEnabled();
    fireEvent.submit(screen.getByLabelText(/Inference data/i).closest("form"));

    expect(await screen.findByText(/ensemble vote, which has no calibrated probability/i)).toBeInTheDocument();
    expect(screen.getByText("n/a")).toBeInTheDocument();
  });

  it("omits the confidence column for regression", async () => {
    api.predictDataset.mockResolvedValue(
      response({
        problem_type: "regression",
        target: "price",
        confidences: null,
        rows: [{ index: 0, prediction: 180432.5 }],
        predictions: [180432.5],
        input_rows: [{ age: 70, region: "north", spend: 51.2 }],
      })
    );
    const user = userEvent.setup();
    renderPanel({ problemType: "regression", target: "price" });

    await user.upload(
      screen.getByLabelText(/Inference data/i),
      new File(["age,region,spend\n70,north,51.2\n"], "p.csv", { type: "text/csv" })
    );
    expect(screen.getByRole("button", { name: /Run predictions/i })).toBeEnabled();
    fireEvent.submit(screen.getByLabelText(/Inference data/i).closest("form"));

    const table = await screen.findByRole("table");
    expect(within(table).queryByText(/Confidence/i)).not.toBeInTheDocument();
  });

  it("shows the API's error message", async () => {
    api.predictDataset.mockRejectedValue(new Error("Missing required features: ['spend']"));
    const user = userEvent.setup();
    renderPanel();

    await user.upload(
      screen.getByLabelText(/Inference data/i),
      new File(["age\n70\n"], "p.csv", { type: "text/csv" })
    );
    expect(screen.getByRole("button", { name: /Run predictions/i })).toBeEnabled();
    fireEvent.submit(screen.getByLabelText(/Inference data/i).closest("form"));

    expect(await screen.findByRole("alert")).toHaveTextContent(/Missing required features/i);
  });
});
