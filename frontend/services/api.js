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

export async function startUploadJob(formData) {
  return unwrap(await fetch(`${API_BASE}/upload`, { method: "POST", body: formData }));
}

export async function getUploadJobStatus(jobId) {
  return unwrap(await fetch(`${API_BASE}/upload/status/${jobId}`));
}

export async function predictDataset(runKey, file) {
  const formData = new FormData();
  formData.append("file", file);
  return unwrap(await fetch(`${API_BASE}/predict/${runKey}`, { method: "POST", body: formData }));
}

export async function chatQuery(payload) {
  return unwrap(
    await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function getInsights(runKey) {
  return unwrap(await fetch(`${API_BASE}/insights/${runKey}`));
}

export async function listRuns(limit = 50) {
  return unwrap(await fetch(`${API_BASE}/runs?limit=${limit}`));
}

export async function deleteRun(runKey) {
  return unwrap(await fetch(`${API_BASE}/runs/${runKey}`, { method: "DELETE" }));
}
