import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ErrorBoundary from "../components/ErrorBoundary";

function Boom() {
  throw new Error("malformed result payload");
}

describe("ErrorBoundary", () => {
  it("contains a render failure instead of blanking the page", () => {
    // React logs the caught error; silence it so the run stays readable.
    vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <div>
        <ErrorBoundary title="The results dashboard could not be displayed">
          <Boom />
        </ErrorBoundary>
        <p>rest of the page</p>
      </div>
    );

    expect(screen.getByText(/results dashboard could not be displayed/i)).toBeInTheDocument();
    expect(screen.getByText(/malformed result payload/i)).toBeInTheDocument();
    // The sibling content is still mounted — that is the whole point.
    expect(screen.getByText("rest of the page")).toBeInTheDocument();
  });

  it("renders children untouched when nothing throws", () => {
    render(
      <ErrorBoundary>
        <p>all good</p>
      </ErrorBoundary>
    );
    expect(screen.getByText("all good")).toBeInTheDocument();
  });
});
