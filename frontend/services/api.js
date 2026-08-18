// Configured at build time. In dev, "/api" is proxied to the backend by
// vite.config.js; in the container it is proxied by nginx. Set VITE_API_BASE
// to an absolute URL when the API lives on a different host.
const API_BASE = import.meta.env?.VITE_API_BASE || "/api";

// The API key, when the backend runs with AUTH_ENABLED. Kept in localStorage
// rather than a cookie because the credential is a bearer token the browser
// must attach deliberately — nothing is sent on a cross-site request that this
// code did not make, which is also why the API refuses credentialed CORS.
const API_KEY_STORAGE = "analyst.apiKey";

export function getApiKey() {
  try {
    return localStorage.getItem(API_KEY_STORAGE) || "";
  } catch {
    // Private-mode Safari and similar. Losing the key means re-entering it,
    // which is better than the whole app failing to render.
    return "";
  }
}

export function setApiKey(key) {
  try {
    const trimmed = (key || "").trim();
    if (trimmed) localStorage.setItem(API_KEY_STORAGE, trimmed);
    else localStorage.removeItem(API_KEY_STORAGE);
  } catch {
    /* the request-level header below still works for this session */
  }
}

export function clearApiKey() {
  setApiKey("");
}

// Thrown so callers can tell "you are not signed in" apart from a real error
// and prompt for a key instead of showing a red banner.
export class UnauthorizedError extends Error {
  constructor(message) {
    super(message);
    this.name = "UnauthorizedError";
    this.status = 401;
  }
}

function withAuth(headers = {}) {
  const key = getApiKey();
  return key ? { ...headers, Authorization: `Bearer ${key}` } : headers;
}

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
  if (res.status === 401) throw new UnauthorizedError(message);
  throw new Error(message);
}

// One place where the credential is attached, so a new endpoint cannot forget.
async function request(path, { headers, ...options } = {}) {
  return unwrap(await fetch(`${API_BASE}${path}`, { ...options, headers: withAuth(headers) }));
}

// Every call forwards `signal`, so an in-flight request is cancelled on
// unmount instead of resolving into a dead component.
export async function startUploadJob(formData, { signal } = {}) {
  return request("/upload", { method: "POST", body: formData, signal });
}

export async function getUploadJobStatus(jobId, { signal } = {}) {
  return request(`/upload/status/${jobId}`, { signal });
}

export async function profileDataset(file, { signal } = {}) {
  const formData = new FormData();
  formData.append("file", file);
  return request("/profile", { method: "POST", body: formData, signal });
}

export async function predictDataset(runKey, file, { signal } = {}) {
  const formData = new FormData();
  formData.append("file", file);
  return request(`/predict/${runKey}`, { method: "POST", body: formData, signal });
}

export async function predictSingleRow(runKey, row, { signal } = {}) {
  return request(`/predict/${runKey}/row`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row }),
    signal,
  });
}

export async function chatQuery(payload, { signal } = {}) {
  return request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export async function generateReport(runKey, { signal } = {}) {
  return request(`/report/${runKey}`, { method: "POST", signal });
}

export async function listConversations(runKey, { signal } = {}) {
  return request(`/chat/${runKey}/conversations`, { signal });
}

export async function getConversation(conversationId, { signal } = {}) {
  return request(`/chat/conversation/${conversationId}`, { signal });
}

export async function getInsights(runKey, { signal } = {}) {
  return request(`/insights/${runKey}`, { signal });
}

export async function listRuns(limit = 50, { signal } = {}) {
  return request(`/runs?limit=${limit}`, { signal });
}

export async function getRunResult(runKey, { signal } = {}) {
  return request(`/runs/${runKey}/result`, { signal });
}

export async function compareRuns(runKeys, { signal } = {}) {
  return request(`/runs/compare?keys=${runKeys.join(",")}`, { signal });
}

export async function deleteRun(runKey, { signal } = {}) {
  return request(`/runs/${runKey}`, { method: "DELETE", signal });
}

// --- account, usage, and data lifecycle (Phase 6) ---------------------------

export async function whoAmI({ signal } = {}) {
  return request("/auth/me", { signal });
}

export async function getUsage(days = 30, { signal } = {}) {
  return request(`/usage?days=${days}`, { signal });
}

export async function getPrivacyPolicy({ signal } = {}) {
  return request("/privacy", { signal });
}

// Irreversible: every dataset, model, and conversation this account owns.
export async function deleteAllMyData({ signal } = {}) {
  return request("/account/data?confirm=true", { method: "DELETE", signal });
}
