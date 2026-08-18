// Configured at build time. In dev, "/api" is proxied to the backend by
// vite.config.js; in the container it is proxied by nginx. Set VITE_API_BASE
// to an absolute URL when the API lives on a different host.
const API_BASE = import.meta.env?.VITE_API_BASE || "/api";

async function unwrap(res) {
  if (res.ok) return res.json();
  // FastAPI errors arrive as {detail: "..."}; show that rather than raw JSON.
  let message = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (body?.detail) message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    try {
      message = (await res.text()) || message;
    } catch {
      /* keep the status-code message */
    }
  }
  throw new Error(message);
}

// Every call forwards `signal`, so an in-flight request is cancelled on
// unmount instead of resolving into a dead component.
export async function startUploadJob(formData, { signal } = {}) {
  return unwrap(await fetch(`${API_BASE}/upload`, { method: "POST", body: formData, signal }));
}

export async function getUploadJobStatus(jobId, { signal } = {}) {
  return unwrap(await fetch(`${API_BASE}/upload/status/${jobId}`, { signal }));
}

export async function profileDataset(file, { signal } = {}) {
  const formData = new FormData();
  formData.append("file", file);
  return unwrap(await fetch(`${API_BASE}/profile`, { method: "POST", body: formData, signal }));
}

export async function predictDataset(runKey, file, { signal } = {}) {
  const formData = new FormData();
  formData.append("file", file);
  return unwrap(await fetch(`${API_BASE}/predict/${runKey}`, { method: "POST", body: formData, signal }));
}

export async function predictSingleRow(runKey, row, { signal } = {}) {
  return unwrap(
    await fetch(`${API_BASE}/predict/${runKey}/row`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ row }),
      signal,
    })
  );
}

export async function chatQuery(payload, { signal } = {}) {
  return unwrap(
    await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    })
  );
}

export async function getInsights(runKey, { signal } = {}) {
  return unwrap(await fetch(`${API_BASE}/insights/${runKey}`, { signal }));
}

export async function listRuns(limit = 50, { signal } = {}) {
  return unwrap(await fetch(`${API_BASE}/runs?limit=${limit}`, { signal }));
}

export async function getRunResult(runKey, { signal } = {}) {
  return unwrap(await fetch(`${API_BASE}/runs/${runKey}/result`, { signal }));
}

export async function compareRuns(runKeys, { signal } = {}) {
  return unwrap(await fetch(`${API_BASE}/runs/compare?keys=${runKeys.join(",")}`, { signal }));
}

export async function deleteRun(runKey, { signal } = {}) {
  return unwrap(await fetch(`${API_BASE}/runs/${runKey}`, { method: "DELETE", signal }));
}
