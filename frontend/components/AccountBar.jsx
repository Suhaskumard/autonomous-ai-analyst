import React, { useCallback, useEffect, useState } from "react";
import { KeyRound, LogOut, ShieldCheck, Wallet } from "lucide-react";

import { clearApiKey, getUsage, setApiKey, whoAmI } from "../services/api";

// Duck-typed rather than `instanceof UnauthorizedError`: the class identity
// does not survive a mocked or duplicated module, and misreading a 401 as a
// hard failure would hide the sign-in prompt behind a generic error.
const isUnauthorized = (error) => error?.status === 401 || error?.name === "UnauthorizedError";

/**
 * The account strip: who you are signed in as, and what you have spent.
 *
 * Renders nothing at all when the backend runs with AUTH_ENABLED=false, which
 * is the default. A single operator on localhost has no account to show, and
 * putting a permanently empty sign-in box above their dashboard would be an
 * invitation to wonder what is broken.
 *
 * When auth *is* on, this is the only way into the app: without a key every
 * request 401s, so the key prompt has to come before anything else can load.
 */
export default function AccountBar({ onStatusChange }) {
  // "unknown" until /auth/me answers — rendering either state before then
  // flashes the wrong UI on every refresh.
  const [status, setStatus] = useState("unknown");
  const [identity, setIdentity] = useState(null);
  const [usage, setUsage] = useState(null);
  const [keyInput, setKeyInput] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(
    async (signal) => {
      try {
        const me = await whoAmI({ signal });
        const next = me.auth_enabled ? "signed-in" : "local";
        setIdentity(me);
        setStatus(next);
        onStatusChange?.({ status: next, identity: me });
        if (me.auth_enabled) {
          // Best effort: a missing usage number is not worth an error banner.
          try {
            setUsage(await getUsage(30, { signal }));
          } catch {
            setUsage(null);
          }
        }
      } catch (err) {
        if (signal?.aborted) return;
        if (isUnauthorized(err)) {
          // A 401 is itself the answer: authentication is on and we lack a key.
          setStatus("signed-out");
          setIdentity(null);
          onStatusChange?.({ status: "signed-out", identity: null });
        } else {
          // The backend being unreachable is the upload form's problem to
          // report, not this strip's.
          setStatus("local");
          onStatusChange?.({ status: "local", identity: null });
        }
      }
    },
    [onStatusChange]
  );

  useEffect(() => {
    const controller = new AbortController();
    load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const signIn = async (event) => {
    event.preventDefault();
    const key = keyInput.trim();
    if (!key) return;
    setError("");
    setApiKey(key);
    try {
      await whoAmI();
      setKeyInput("");
      await load();
    } catch (err) {
      // Do not keep a key that does not work; leaving it stored makes every
      // later request fail for a reason the user has already been told about.
      clearApiKey();
      setError(isUnauthorized(err) ? "That API key was not accepted." : err.message);
    }
  };

  const signOut = () => {
    clearApiKey();
    setIdentity(null);
    setUsage(null);
    setStatus("signed-out");
    onStatusChange?.({ status: "signed-out", identity: null });
  };

  if (status === "unknown" || status === "local") return null;

  if (status === "signed-out") {
    return (
      <form className="account-bar account-bar-signin" onSubmit={signIn}>
        <div className="account-identity">
          <KeyRound size={16} />
          <span>This server requires an API key.</span>
        </div>
        <div className="account-actions">
          <input
            className="input account-key-input"
            type="password"
            value={keyInput}
            onChange={(event) => setKeyInput(event.target.value)}
            placeholder="analyst_..."
            aria-label="API key"
            autoComplete="off"
          />
          <button className="btn btn-primary btn-compact" type="submit" disabled={!keyInput.trim()}>
            Use key
          </button>
        </div>
        {error && (
          <p className="account-error" role="alert">
            {error}
          </p>
        )}
      </form>
    );
  }

  return (
    <div className="account-bar">
      <div className="account-identity">
        <ShieldCheck size={16} />
        <span>{identity?.email}</span>
        {identity?.is_admin && <span className="account-tag">admin</span>}
      </div>
      <div className="account-actions">
        {usage && (
          // Labelled an estimate wherever it appears: the number comes from
          // published list prices, not from the provider's billing.
          <span className="account-usage" title={`${usage.calls} model call(s) in the last ${usage.window_days} days`}>
            <Wallet size={14} />
            est. ${usage.estimated_cost_usd?.toFixed(4)} / {usage.window_days}d
          </span>
        )}
        <button className="btn btn-secondary btn-compact" type="button" onClick={signOut}>
          <LogOut size={14} />
          Sign out
        </button>
      </div>
    </div>
  );
}
