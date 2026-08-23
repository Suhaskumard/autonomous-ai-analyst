import React, { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

const STORAGE_KEY = "aaa-theme";

function readStoredTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : "";
  } catch {
    // Private browsing or storage disabled — fall back to the system preference.
    return "";
  }
}

function systemPrefersLight() {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: light)").matches;
}

/**
 * Light/dark toggle.
 *
 * The app is dark by default and respects `prefers-color-scheme: light` on
 * its own (see styles.css) — this only needs to track an explicit override
 * once the user picks one, so a light-mode system doesn't get stuck showing
 * dark just because someone clicked the toggle once in a previous session.
 */
export default function ThemeToggle() {
  const [theme, setTheme] = useState(readStoredTheme);

  useEffect(() => {
    if (theme) {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }, [theme]);

  const isLight = theme ? theme === "light" : systemPrefersLight();

  const toggle = () => {
    const next = isLight ? "dark" : "light";
    setTheme(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* preference just won't persist across a reload */
    }
  };

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      aria-label={isLight ? "Switch to dark theme" : "Switch to light theme"}
      title={isLight ? "Switch to dark theme" : "Switch to light theme"}
    >
      {isLight ? <Moon size={17} /> : <Sun size={17} />}
    </button>
  );
}
