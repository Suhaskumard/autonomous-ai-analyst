import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// jsdom does not implement scrollIntoView, which ChatBox calls on every render.
Element.prototype.scrollIntoView = vi.fn();

// Recharts' ResponsiveContainer observes its box; jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver || ResizeObserverStub;

// …and every element it measures reports 0×0 in jsdom, so the container never
// renders its children. Give layout boxes a non-zero size for the tests.
const boundingRect = {
  width: 800,
  height: 400,
  top: 0,
  left: 0,
  bottom: 400,
  right: 800,
  x: 0,
  y: 0,
  toJSON: () => {},
};
Element.prototype.getBoundingClientRect = vi.fn(() => boundingRect);
Object.defineProperties(HTMLElement.prototype, {
  offsetWidth: { configurable: true, get: () => 800 },
  offsetHeight: { configurable: true, get: () => 400 },
});
