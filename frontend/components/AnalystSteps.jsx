import React, { useState } from "react";
import { AlertTriangle, ChevronDown, ChevronRight, Code2, Image as ImageIcon, Table2, Wrench } from "lucide-react";

import { STATUS } from "./charts/theme";

const TOOL_LABEL = {
  describe_column: "Described a column",
  correlate: "Computed correlations",
  filter_rows: "Filtered rows",
  group_stats: "Aggregated by group",
  plot: "Drew a chart",
  run_model: "Ran the model",
  run_python: "Executed Python",
};

/**
 * The work behind an answer, rendered as what it was.
 *
 * The old chat collapsed everything to one markdown blob, so a table arrived
 * as pipe characters, a chart arrived as a sentence describing it, and the
 * code that produced a number was not shown at all. Each step here keeps its
 * own shape — that is what makes an answer checkable rather than merely
 * plausible.
 */
export default function AnalystSteps({ steps }) {
  if (!steps || steps.length === 0) return null;
  return (
    <div className="analyst-steps">
      {steps.map((step, index) => (
        <Step key={index} step={step} index={index} />
      ))}
    </div>
  );
}

function Step({ step, index }) {
  const [open, setOpen] = useState(false);
  const label = TOOL_LABEL[step.tool] || step.tool;
  const code = step.code || step.execution?.code;
  const figures = step.figures || step.execution?.figures || [];
  const table = step.table || step.matrix || step.execution?.result;
  const stdout = step.execution?.stdout;
  const hasDetail = Boolean(code || figures.length || (table && table.kind === "table") || stdout);

  return (
    <div className={step.ok ? "analyst-step" : "analyst-step analyst-step-failed"}>
      <button
        type="button"
        className="analyst-step-head"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        disabled={!hasDetail}
      >
        {hasDetail ? (
          open ? (
            <ChevronDown size={13} />
          ) : (
            <ChevronRight size={13} />
          )
        ) : (
          <span className="analyst-step-bullet" aria-hidden="true" />
        )}
        {step.ok ? <Wrench size={13} /> : <AlertTriangle size={13} style={{ color: STATUS.critical }} />}
        <span className="analyst-step-label">
          {index + 1}. {label}
        </span>
        <span className="analyst-step-summary">{step.summary}</span>
      </button>

      {open && hasDetail && (
        <div className="analyst-step-body">
          {code && (
            <figure className="analyst-code">
              <figcaption>
                <Code2 size={12} /> Code that produced this
              </figcaption>
              <pre>
                <code>{code}</code>
              </pre>
            </figure>
          )}

          {stdout && stdout.trim() && (
            <figure className="analyst-code">
              <figcaption>Output</figcaption>
              <pre>
                <code>{stdout}</code>
              </pre>
            </figure>
          )}

          {table && table.kind === "table" && <ResultTable table={table} />}

          {figures.map((figure, position) => (
            <figure key={position} className="analyst-figure">
              <figcaption>
                <ImageIcon size={12} /> {figure.caption || "Chart"}
              </figcaption>
              <img src={`data:${figure.mime || "image/png"};base64,${figure.data}`} alt={figure.caption || "Chart"} />
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}

function ResultTable({ table }) {
  return (
    <figure className="analyst-table">
      <figcaption>
        <Table2 size={12} />
        {table.shape ? ` ${table.shape[0]} rows × ${table.shape[1]} cols` : " Result"}
        {table.truncated && " (truncated)"}
      </figcaption>
      <div className="chart-table-wrap">
        <table className="chart-table">
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th key={column} scope="col">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.slice(0, 25).map((row, index) => (
              <tr key={index}>
                {row.map((cell, position) => (
                  <td key={position} className="tabular">
                    {cell === null || cell === undefined ? "—" : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </figure>
  );
}
