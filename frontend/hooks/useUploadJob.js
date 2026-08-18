import { useCallback, useEffect, useRef, useState } from "react";

import { getUploadJobStatus, startUploadJob } from "../services/api";

/**
 * Drives upload → poll → result.
 *
 * The previous version was a bare `while` loop with a fixed 1s sleep, no
 * unmount cleanup, no abort, and no timeout — so it kept polling and calling
 * setState after the component was gone, forever, if a job never finished.
 * Everything here is cancellable and bounded.
 */

export const POLL_INITIAL_MS = 700;
export const POLL_MAX_MS = 5000;
export const POLL_BACKOFF = 1.35;
export const POLL_TIMEOUT_MS = 15 * 60 * 1000;

export function nextDelay(current) {
  return Math.min(Math.round(current * POLL_BACKOFF), POLL_MAX_MS);
}

const sleep = (ms, signal) =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true }
    );
  });

export default function useUploadJob() {
  const [state, setState] = useState({
    loading: false,
    status: null,
    result: null,
    error: "",
    errorKind: "",
  });

  // One controller per run, aborted on unmount or when a new run starts.
  const controllerRef = useRef(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  const update = useCallback((patch) => {
    // The guard is what stops "setState on an unmounted component".
    if (mountedRef.current) setState((current) => ({ ...current, ...patch }));
  }, []);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    update({ loading: false });
  }, [update]);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setState({ loading: false, status: null, result: null, error: "", errorKind: "" });
  }, []);

  const showResult = useCallback((result) => {
    setState({ loading: false, status: null, result, error: "", errorKind: "" });
  }, []);

  const run = useCallback(
    async (formData) => {
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;
      const { signal } = controller;

      setState({ loading: true, status: null, result: null, error: "", errorKind: "" });

      try {
        const started = await startUploadJob(formData, { signal });
        const jobId = started.job_id;
        const deadline = Date.now() + POLL_TIMEOUT_MS;
        let delay = POLL_INITIAL_MS;

        // eslint-disable-next-line no-constant-condition
        while (true) {
          if (signal.aborted) return;
          if (Date.now() > deadline) {
            update({
              loading: false,
              error: "Timed out waiting for this job. It may still be running — check the workspace.",
              errorKind: "timeout",
            });
            return;
          }

          const status = await getUploadJobStatus(jobId, { signal });
          update({ status });

          if (status.state === "completed") {
            update({ loading: false, result: status.result });
            return;
          }
          if (status.state === "failed") {
            update({
              loading: false,
              error: status.error || "Job failed.",
              errorKind: status.error_kind || "pipeline",
            });
            return;
          }

          await sleep(delay, signal);
          // Back off so a long training run is not polled 900 times.
          delay = nextDelay(delay);
        }
      } catch (error) {
        if (error?.name === "AbortError") return; // deliberate cancel/unmount
        update({ loading: false, error: error?.message || "Failed to process dataset.", errorKind: "network" });
      }
    },
    [update]
  );

  return { ...state, run, cancel, reset, showResult };
}
