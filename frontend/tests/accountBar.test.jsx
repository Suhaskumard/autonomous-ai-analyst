import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../services/api", () => ({
  whoAmI: vi.fn(),
  getUsage: vi.fn(),
  setApiKey: vi.fn(),
  clearApiKey: vi.fn(),
}));

import AccountBar from "../components/AccountBar";
import * as api from "../services/api";

function unauthorized() {
  const error = new Error("Authentication required.");
  error.name = "UnauthorizedError";
  error.status = 401;
  return error;
}

describe("account strip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getUsage.mockResolvedValue({ calls: 3, estimated_cost_usd: 0.0123, window_days: 30 });
  });

  it("stays out of the way on a single-operator install", async () => {
    api.whoAmI.mockResolvedValue({ user_id: "local", email: "local@localhost", auth_enabled: false });

    const { container } = render(<AccountBar />);

    // Nothing to sign in to, so nothing is drawn — not an empty sign-in box.
    await waitFor(() => expect(api.whoAmI).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it("asks for a key when the server rejects an unauthenticated request", async () => {
    api.whoAmI.mockRejectedValue(unauthorized());

    render(<AccountBar />);

    expect(await screen.findByText(/requires an API key/i)).toBeInTheDocument();
  });

  it("reports the signed-in account and its estimated spend", async () => {
    api.whoAmI.mockResolvedValue({ user_id: "u1", email: "alice@example.com", is_admin: true, auth_enabled: true });

    render(<AccountBar />);

    expect(await screen.findByText("alice@example.com")).toBeInTheDocument();
    expect(screen.getByText("admin")).toBeInTheDocument();
    // Always labelled an estimate, because it is one.
    expect(screen.getByText(/est\. \$0\.0123/)).toBeInTheDocument();
  });

  it("keeps a key that works", async () => {
    api.whoAmI.mockRejectedValueOnce(unauthorized());
    render(<AccountBar />);
    await screen.findByText(/requires an API key/i);

    api.whoAmI.mockResolvedValue({ user_id: "u1", email: "alice@example.com", auth_enabled: true });
    fireEvent.change(screen.getByLabelText(/api key/i), { target: { value: "analyst_good" } });
    fireEvent.click(screen.getByRole("button", { name: /use key/i }));

    expect(await screen.findByText("alice@example.com")).toBeInTheDocument();
    expect(api.setApiKey).toHaveBeenCalledWith("analyst_good");
    expect(api.clearApiKey).not.toHaveBeenCalled();
  });

  it("discards a key that does not work, rather than failing every later call with it", async () => {
    api.whoAmI.mockRejectedValue(unauthorized());
    render(<AccountBar />);
    await screen.findByText(/requires an API key/i);

    fireEvent.change(screen.getByLabelText(/api key/i), { target: { value: "analyst_bad" } });
    fireEvent.click(screen.getByRole("button", { name: /use key/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/not accepted/i);
    expect(api.clearApiKey).toHaveBeenCalled();
  });

  it("signing out drops the stored credential", async () => {
    api.whoAmI.mockResolvedValue({ user_id: "u1", email: "alice@example.com", auth_enabled: true });
    render(<AccountBar />);
    await screen.findByText("alice@example.com");

    fireEvent.click(screen.getByRole("button", { name: /sign out/i }));

    expect(api.clearApiKey).toHaveBeenCalled();
    expect(await screen.findByText(/requires an API key/i)).toBeInTheDocument();
  });
});
