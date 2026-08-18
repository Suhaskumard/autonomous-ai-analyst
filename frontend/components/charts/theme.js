/**
 * Chart tokens.
 *
 * The app renders on one dark surface (#0f172a card over #030712), so these
 * are the dark steps of the reference palette, validated against *this*
 * surface rather than assumed:
 *
 *   validate_palette.js "#3987e5,#d95926,#199e70,#c98500" --mode dark --surface "#0f172a"
 *     lightness band PASS · chroma floor PASS · CVD separation PASS (worst
 *     adjacent ΔE 8.4) · normal-vision floor PASS (19.8) · contrast PASS
 *
 * The same four under `--pairs all` FAIL (yellow↔orange, ΔE 4.8 deutan), and
 * the first three PASS. Hence the two caps below: bars and lines may use four
 * series, anything where every pair can sit side by side uses three.
 */

export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"];

/** Adjacent-pair contexts: grouped/stacked bars, lines. */
export const MAX_SERIES_ADJACENT = 4;
/** All-pairs contexts: scatter, categorical cells. */
export const MAX_SERIES_ALL_PAIRS = 3;

/** Single-series default — most charts here plot one measure. */
export const PRIMARY = SERIES[0];

/** Sequential ramp (one hue, light→dark) for magnitude: heatmap cells. */
export const SEQUENTIAL = [
  "#cde2fb",
  "#9ec5f4",
  "#6da7ec",
  "#3987e5",
  "#256abf",
  "#184f95",
  "#0d366b",
];

/** Diverging pair for polarity (correlation): two hues + a neutral midpoint. */
export const DIVERGING = {
  negative: "#d03b3b",
  neutral: "#383835",
  positive: "#3987e5",
};

/** Status colours are reserved — never reused as a series colour. */
export const STATUS = {
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
};

export const VERDICT_COLOR = {
  strong: STATUS.good,
  fair: STATUS.warning,
  unreliable: STATUS.critical,
};

/** Chart chrome: recessive by design, so the marks carry the eye. */
export const INK = {
  primary: "#f8fafc",
  secondary: "#94a3b8",
  muted: "#64748b",
  grid: "#1e293b",
  axis: "#334155",
  surface: "#0f172a",
};

export const AXIS_PROPS = {
  stroke: INK.axis,
  tick: { fill: INK.muted, fontSize: 11 },
  tickLine: false,
};

export const GRID_PROPS = {
  stroke: INK.grid,
  strokeDasharray: "3 3",
  vertical: false,
};

/** Shared tooltip chrome, so every chart's hover layer looks like one system. */
export const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#0b1220",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: "10px",
    fontSize: "0.8rem",
    color: INK.primary,
    boxShadow: "0 10px 30px rgba(0,0,0,0.45)",
  },
  labelStyle: { color: INK.secondary, marginBottom: 4 },
  itemStyle: { color: INK.primary },
  cursor: { fill: "rgba(255,255,255,0.05)" },
};

/** 4px rounded data-end, anchored to the baseline. */
export const BAR_RADIUS = [4, 4, 0, 0];
export const BAR_RADIUS_HORIZONTAL = [0, 4, 4, 0];

export function seriesColor(index) {
  return SERIES[index % SERIES.length];
}

/** Map a value in [0,1] onto the sequential ramp. */
export function sequentialColor(fraction) {
  if (!Number.isFinite(fraction)) return SEQUENTIAL[0];
  const clamped = Math.min(Math.max(fraction, 0), 1);
  return SEQUENTIAL[Math.round(clamped * (SEQUENTIAL.length - 1))];
}

/**
 * Map a correlation in [-1,1] onto the diverging pair.
 * Gray at zero: "no relationship" must not look like a weak positive one.
 */
export function divergingColor(value) {
  if (!Number.isFinite(value)) return DIVERGING.neutral;
  const magnitude = Math.min(Math.abs(value), 1);
  const target = value >= 0 ? DIVERGING.positive : DIVERGING.negative;
  return mix(DIVERGING.neutral, target, magnitude);
}

function mix(fromHex, toHex, amount) {
  const from = hexToRgb(fromHex);
  const to = hexToRgb(toHex);
  const channel = (a, b) => Math.round(a + (b - a) * amount);
  return `rgb(${channel(from[0], to[0])}, ${channel(from[1], to[1])}, ${channel(from[2], to[2])})`;
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16),
  ];
}

/** Compact number formatting for axes and labels. */
export function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  if (Number.isInteger(value)) return String(value);
  if (abs < 0.01 && abs > 0) return value.toExponential(1);
  return value.toFixed(digits);
}
