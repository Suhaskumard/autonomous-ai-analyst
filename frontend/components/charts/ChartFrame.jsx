import React, { useState } from "react";
import { Table2, BarChart3 } from "lucide-react";

import { INK } from "./theme";

/**
 * The chrome every chart shares: title, caption, and a table view.
 *
 * The table view is the accessibility fallback the palette's contrast relief
 * rule requires — every chart's numbers are readable without relying on colour.
 */
export default function ChartFrame({ title, caption, tableHeaders, tableRows, children, height = 260 }) {
  const [showTable, setShowTable] = useState(false);
  const hasTable = Array.isArray(tableRows) && tableRows.length > 0;

  return (
    <section className="chart-frame">
      <header className="chart-frame-head">
        <div>
          <h3 className="chart-title">{title}</h3>
          {caption && <p className="chart-caption">{caption}</p>}
        </div>
        {hasTable && (
          <button
            type="button"
            className="chart-toggle"
            onClick={() => setShowTable((current) => !current)}
            aria-pressed={showTable}
          >
            {showTable ? <BarChart3 size={14} /> : <Table2 size={14} />}
            {showTable ? "Chart" : "Table"}
          </button>
        )}
      </header>

      {showTable && hasTable ? (
        <div className="chart-table-wrap" style={{ maxHeight: height + 40 }}>
          <table className="chart-table">
            <thead>
              <tr>
                {tableHeaders.map((header) => (
                  <th key={header} scope="col">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((row, index) => (
                <tr key={index}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ width: "100%", height }}>{children}</div>
      )}
    </section>
  );
}

export function EmptyChart({ message }) {
  return (
    <div
      style={{
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: INK.muted,
        fontSize: "0.85rem",
        textAlign: "center",
        padding: "0 24px",
      }}
    >
      {message}
    </div>
  );
}
