import React, { useCallback, useEffect, useState } from "react";
import { Calendar, Check, Copy, Download, Link2, Loader2, Package, Trash2 } from "lucide-react";

import {
  createReportSchedule,
  createShareLink,
  deleteReportSchedule,
  downloadExport,
  getExportAvailability,
  listReportSchedules,
  listShareLinks,
  revokeShareLink,
} from "../services/api";

/**
 * What you can do with a finished run besides look at it — Phase 10.
 *
 * Three capabilities, each of which the server may have switched off, and
 * each of which says so rather than failing when it has. Sharing and
 * scheduling are opt-in server-side (SHARE_LINKS_ENABLED,
 * SCHEDULED_REPORTS_ENABLED); export is always available in at least the
 * bundle format.
 */
export default function RunActions({ runKey }) {
  return (
    <div className="card fade-in">
      <h2 style={{ margin: "0 0 4px", display: "flex", alignItems: "center", gap: 10, fontSize: "1.15rem" }}>
        <Package size={20} style={{ color: "var(--primary)" }} />
        Share, export, and schedule
      </h2>
      <p style={{ color: "var(--text-dim)", fontSize: "0.85rem", margin: "0 0 20px" }}>
        Hand this run to someone else, take the model with you, or have the report arrive on its own.
      </p>
      <div className="run-actions">
        <ShareSection runKey={runKey} />
        <ExportSection runKey={runKey} />
        <ScheduleSection runKey={runKey} />
      </div>
    </div>
  );
}

function ShareSection({ runKey }) {
  const [links, setLinks] = useState([]);
  const [newToken, setNewToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const body = await listShareLinks(runKey);
      setLinks(body.links || []);
    } catch {
      /* listing is a nicety; the create button reports the real error */
    }
  }, [runKey]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const create = async () => {
    setBusy(true);
    setError("");
    setCopied(false);
    try {
      const created = await createShareLink(runKey);
      setNewToken(created.token);
      await refresh();
    } catch (err) {
      setError(err?.message || "Could not create a share link.");
    } finally {
      setBusy(false);
    }
  };

  const shareUrl = newToken ? `${window.location.origin}/share/${newToken}` : "";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
    } catch {
      setError("Could not copy — select the link and copy it by hand.");
    }
  };

  const activeCount = links.filter((link) => !link.revoked_at).length;

  return (
    <section className="run-action">
      <h3>
        <Link2 size={15} />
        Read-only link
      </h3>
      <p className="run-action-note">
        Anyone with the link sees this dashboard and nothing else — no chat, no predictions, no report
        generation. It expires on its own.
      </p>

      <button type="button" className="btn btn-secondary btn-compact" onClick={create} disabled={busy}>
        {busy ? <Loader2 size={13} className="animate-spin" /> : <Link2 size={13} />}
        Create link
      </button>

      {newToken && (
        <div className="run-action-result">
          {/* Shown once, like an API key: the server stores no copy it could
              give back, so this is the only chance to take it. */}
          <p className="run-action-warn">Copy this now — it is not shown again.</p>
          <div className="share-link-row">
            <input className="input" value={shareUrl} readOnly onFocus={(event) => event.target.select()} />
            <button type="button" className="btn btn-secondary btn-compact" onClick={copy}>
              {copied ? <Check size={13} /> : <Copy size={13} />}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
      )}

      {activeCount > 0 && (
        <p className="run-action-note">
          {activeCount} active link{activeCount === 1 ? "" : "s"} for this run.{" "}
          {newToken && (
            <button
              type="button"
              className="link-button"
              onClick={async () => {
                await revokeShareLink(newToken);
                setNewToken("");
                await refresh();
              }}
            >
              Revoke the one above
            </button>
          )}
        </p>
      )}

      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}

function ExportSection({ runKey }) {
  const [availability, setAvailability] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    getExportAvailability(runKey)
      .then((body) => {
        if (!cancelled) setAvailability(body);
      })
      .catch(() => {
        if (!cancelled) setAvailability({ bundle: { available: true }, onnx: { available: false, reason: "" } });
      });
    return () => {
      cancelled = true;
    };
  }, [runKey]);

  const download = async (format, filename) => {
    setBusy(format);
    setError("");
    try {
      const blob = await downloadExport(runKey, format);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err?.message || "The export could not be downloaded.");
    } finally {
      setBusy("");
    }
  };

  const onnxAvailable = availability?.onnx?.available;

  return (
    <section className="run-action">
      <h3>
        <Download size={15} />
        Export the model
      </h3>
      <p className="run-action-note">
        Take the fitted pipeline somewhere this application is not installed.
      </p>

      <button
        type="button"
        className="btn btn-secondary btn-compact"
        onClick={() => download("bundle", `${runKey}_export.zip`)}
        disabled={busy === "bundle"}
      >
        {busy === "bundle" ? <Loader2 size={13} className="animate-spin" /> : <Package size={13} />}
        Portable bundle (.zip)
      </button>

      <button
        type="button"
        className="btn btn-secondary btn-compact"
        onClick={() => download("onnx", `${runKey}_model.onnx`)}
        disabled={!onnxAvailable || busy === "onnx"}
        title={onnxAvailable ? "" : availability?.onnx?.reason || "Checking…"}
      >
        {busy === "onnx" ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
        ONNX
      </button>

      {availability && !onnxAvailable && availability.onnx?.reason && (
        <p className="run-action-note run-action-dim">{availability.onnx.reason}</p>
      )}

      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}

function ScheduleSection({ runKey }) {
  const [schedules, setSchedules] = useState([]);
  const [recipients, setRecipients] = useState("");
  const [interval, setInterval] = useState("weekly");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const body = await listReportSchedules(runKey);
      setSchedules(body.schedules || []);
    } catch {
      /* the create button reports the real error, including "disabled" */
    }
  }, [runKey]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const create = async () => {
    setBusy(true);
    setError("");
    try {
      await createReportSchedule(
        runKey,
        recipients.split(",").map((address) => address.trim()).filter(Boolean),
        interval
      );
      setRecipients("");
      await refresh();
    } catch (err) {
      setError(err?.message || "Could not create that schedule.");
    } finally {
      setBusy(false);
    }
  };

  const remove = async (scheduleId) => {
    setError("");
    try {
      await deleteReportSchedule(scheduleId);
      await refresh();
    } catch (err) {
      setError(err?.message || "Could not delete that schedule.");
    }
  };

  return (
    <section className="run-action">
      <h3>
        <Calendar size={15} />
        Scheduled report
      </h3>
      <p className="run-action-note">
        The full report, regenerated and emailed on a cadence. Delivery is driven by the server&apos;s cron, not
        by this page staying open.
      </p>

      <div className="input-group" style={{ margin: 0 }}>
        <label htmlFor={`recipients-${runKey}`}>Recipients (comma-separated)</label>
        <input
          id={`recipients-${runKey}`}
          className="input"
          type="text"
          value={recipients}
          placeholder="alex@example.com, sam@example.com"
          onChange={(event) => setRecipients(event.target.value)}
        />
      </div>

      <div className="input-group" style={{ margin: 0 }}>
        <label htmlFor={`interval-${runKey}`}>Cadence</label>
        <select
          id={`interval-${runKey}`}
          className="input"
          value={interval}
          onChange={(event) => setInterval(event.target.value)}
        >
          <option value="daily">Daily</option>
          <option value="weekly">Weekly</option>
        </select>
      </div>

      <button
        type="button"
        className="btn btn-secondary btn-compact"
        onClick={create}
        disabled={busy || !recipients.trim()}
      >
        {busy ? <Loader2 size={13} className="animate-spin" /> : <Calendar size={13} />}
        Schedule it
      </button>

      {schedules.length > 0 && (
        <ul className="schedule-list">
          {schedules.map((schedule) => (
            <li key={schedule.schedule_id}>
              <span>
                {schedule.interval} → {schedule.recipients.join(", ")}
                {schedule.last_status && <span className="run-action-dim"> · last: {schedule.last_status}</span>}
              </span>
              <button
                type="button"
                className="btn btn-danger btn-compact"
                onClick={() => remove(schedule.schedule_id)}
                aria-label={`Delete schedule ${schedule.schedule_id}`}
              >
                <Trash2 size={12} />
              </button>
            </li>
          ))}
        </ul>
      )}

      {error && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}
