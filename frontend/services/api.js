const API_BASE = "http://localhost:8000/api";

export async function startUploadJob(formData) {
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getUploadJobStatus(jobId) {
  const res = await fetch(`${API_BASE}/upload/status/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predictDataset(datasetHash, file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/predict/${datasetHash}`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function chatQuery(payload) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
