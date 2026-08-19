import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../services/api", () => ({
  listShareLinks: vi.fn(() => Promise.resolve({ links: [] })),
  createShareLink: vi.fn(),
  revokeShareLink: vi.fn(),
  getExportAvailability: vi.fn(),
  downloadExport: vi.fn(),
  listReportSchedules: vi.fn(() => Promise.resolve({ schedules: [] })),
  createReportSchedule: vi.fn(),
  deleteReportSchedule: vi.fn(),
}));

import RunActions from "../components/RunActions";
import * as api from "../services/api";

const RUN_KEY = "b".repeat(64);

function bundleOnly() {
  return {
    bundle: { available: true, reason: null },
    onnx: { available: false, reason: "This run has categorical feature(s) (region)." },
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  api.listShareLinks.mockResolvedValue({ links: [] });
  api.listReportSchedules.mockResolvedValue({ schedules: [] });
  api.getExportAvailability.mockResolvedValue(bundleOnly());
});

/** Render, then let all three sections' on-mount fetches settle. */
async function renderActions() {
  const view = render(<RunActions runKey={RUN_KEY} />);
  await act(async () => {});
  return view;
}

describe("share links", () => {
  it("shows the created link once, with a warning that it is not shown again", async () => {
    api.createShareLink.mockResolvedValue({ token: "tok_abc123", expires_at: "2026-09-01T00:00:00" });
    await renderActions();

    fireEvent.click(screen.getByRole("button", { name: /create link/i }));

    await waitFor(() => expect(screen.getByDisplayValue(/tok_abc123/)).toBeInTheDocument());
    expect(screen.getByText(/not shown again/i)).toBeInTheDocument();
  });

  it("reports a server that has sharing switched off rather than failing silently", async () => {
    api.createShareLink.mockRejectedValue(new Error("Sharing is disabled on this server."));
    await renderActions();

    fireEvent.click(screen.getByRole("button", { name: /create link/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/sharing is disabled/i));
  });
});

describe("export", () => {
  it("disables ONNX and says why when the pipeline cannot be converted", async () => {
    await renderActions();

    await waitFor(() => expect(screen.getByRole("button", { name: /^onnx$/i })).toBeDisabled());
    expect(screen.getByText(/categorical feature/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /portable bundle/i })).toBeEnabled();
  });

  it("enables ONNX when the server says the run supports it", async () => {
    api.getExportAvailability.mockResolvedValue({
      bundle: { available: true, reason: null },
      onnx: { available: true, reason: null },
    });
    await renderActions();

    await waitFor(() => expect(screen.getByRole("button", { name: /^onnx$/i })).toBeEnabled());
  });
});

describe("scheduled reports", () => {
  it("splits comma-separated recipients before sending them", async () => {
    api.createReportSchedule.mockResolvedValue({ schedule_id: "sched_1" });
    await renderActions();

    fireEvent.change(screen.getByLabelText(/recipients/i), {
      target: { value: "alex@example.com, sam@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: /schedule it/i }));

    await waitFor(() =>
      expect(api.createReportSchedule).toHaveBeenCalledWith(
        RUN_KEY,
        ["alex@example.com", "sam@example.com"],
        "weekly"
      )
    );
  });

  it("cannot be submitted with no recipients", async () => {
    await renderActions();
    expect(screen.getByRole("button", { name: /schedule it/i })).toBeDisabled();
  });

  it("lists existing schedules with their last status", async () => {
    api.listReportSchedules.mockResolvedValue({
      schedules: [
        {
          schedule_id: "sched_1",
          interval: "daily",
          recipients: ["alex@example.com"],
          last_status: "sent",
          next_run_at: "2026-08-20T00:00:00",
        },
      ],
    });
    await renderActions();

    await waitFor(() => expect(screen.getByText(/alex@example.com/)).toBeInTheDocument());
    expect(screen.getByText(/last: sent/i)).toBeInTheDocument();
  });
});
