import React, { useState } from "react";
import { ChevronDown, ChevronRight, ScrollText, Scale, SlidersHorizontal } from "lucide-react";

import { STATUS, formatNumber } from "./charts/theme";

const METRIC_LABEL = {
  f1_macro: "macro F1",
  rmse: "RMSE",
  accuracy: "accuracy",
  balanced_accuracy: "balanced accuracy",
};

/**
 * How the number was produced, not just what it was.
 *
 * "RandomForest, 0.87" is uninterpretable three months later: it says nothing
 * about which columns went in, whether the classes were balanced, what the
 * tuner changed, or which scikit-learn produced it. Everything here is already
 * computed by the run; this is where it becomes readable.
 */
export default function ModelCard({ result }) {
  const card = result?.model_card;
  if (!card) return null;

  const training = card.training || {};
  const data = card.data || {};
  const metric = result.selection_metric || training.selection_metric;
  const label = METRIC_LABEL[metric] || metric;
  const cv = (result.cv_scores?.[result.selected_model] || {})[metric];
  const holdout = result.all_model_scores?.[result.selected_model]?.[metric];

  return (
    <div className="card fade-in">
      <h2>
        <ScrollText size={20} style={{ color: "var(--primary)" }} />
        Model card
      </h2>
      <p className="card-lede">
        Selected on cross-validated <strong>{label}</strong> — not on accuracy, and not on a single split.
      </p>

      <dl className="facts-grid">
        <Fact label={`Cross-validated ${label}`}>
          {cv ? (
            <span className="tabular">
              {formatNumber(cv.mean, 4)} ± {formatNumber(cv.std, 4)}
            </span>
          ) : (
            "—"
          )}
          <span className="fact-note"> over {result.cv_folds ?? training.cv_folds ?? "—"} folds</span>
        </Fact>
        <Fact label={`Held-out ${label}`}>
          <span className="tabular">{holdout === undefined ? "—" : formatNumber(holdout, 4)}</span>
          <span className="fact-note"> on {(result.holdout_rows ?? 0).toLocaleString()} unseen rows</span>
        </Fact>
        <Fact label="Candidates tried">{(training.candidates || []).length || "—"}</Fact>
        <Fact label="Selection metric">{label}</Fact>
        <Fact label="Class weighting">
          {/* Every model gets the same strategy, so listing it once per model
              is seven repetitions of one fact. Group by strategy instead. */}
          {summariseWeighting(result.class_weighting)}
        </Fact>
        <Fact label="Oversampling">{result.smote_used ? "SMOTE (training folds only)" : "none"}</Fact>
        <Fact label="Tuning budget">
          {result.tuning_budget_seconds ? `${result.tuning_budget_seconds}s` : "none — defaults used"}
        </Fact>
        <Fact label="Encoding">
          {`${(data.features?.numeric || []).length} numeric · ${(data.features?.one_hot || []).length} one-hot · ${
            (data.features?.target_encoded || []).length
          } target-encoded`}
        </Fact>
      </dl>

      {(result.ensemble_members || []).length > 0 && (
        <p className="dropped-note">
          <Scale size={13} />
          <span>
            Ensemble of <strong>{result.ensemble_members.join(", ")}</strong>, weighted by how far each beat the
            naive baseline. Models that did not beat it were left out of the vote entirely.
          </span>
        </p>
      )}

      <ClassBalance imbalance={result.imbalance} />
      <PerClassTable rows={result.per_class} />
      <TuningTable reports={result.tuning} />
      <Libraries versions={card.environment?.libraries} />
    </div>
  );
}

function Fact({ label, children }) {
  return (
    <div className="fact">
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

/** "class_weight=balanced (6 models), scale_pos_weight=7.82 (XGBoost)". */
function summariseWeighting(weighting) {
  const entries = Object.entries(weighting || {});
  if (entries.length === 0) return "not needed";

  const byStrategy = new Map();
  entries.forEach(([model, strategy]) => {
    byStrategy.set(strategy, [...(byStrategy.get(strategy) || []), model]);
  });

  return Array.from(byStrategy.entries())
    .map(([strategy, models]) => `${strategy} (${models.length === 1 ? models[0] : `${models.length} models`})`)
    .join(", ");
}

/** The class distribution, as a bar you can see the skew in. */
function ClassBalance({ imbalance }) {
  const distribution = imbalance?.distribution || [];
  if (distribution.length === 0) return null;

  return (
    <div className="summary-table-block">
      <h3 className="summary-table-caption">Class balance</h3>
      <div className="balance-bar" role="img" aria-label={distribution.map((d) => `${d.label} ${Math.round(d.share * 100)}%`).join(", ")}>
        {distribution.map((entry, index) => (
          <div
            key={entry.label}
            className="balance-segment"
            style={{
              width: `${entry.share * 100}%`,
              background: `var(--primary)`,
              opacity: 1 - index * (0.6 / Math.max(distribution.length - 1, 1)),
            }}
          />
        ))}
      </div>
      <ul className="balance-legend">
        {distribution.map((entry) => (
          <li key={entry.label}>
            <span className="balance-legend-label">{entry.label}</span>
            <span className="tabular">
              {entry.count.toLocaleString()} ({(entry.share * 100).toFixed(1)}%)
            </span>
          </li>
        ))}
      </ul>
      {(imbalance.warnings || []).map((warning, index) => (
        <p key={index} className="balance-warning" style={{ color: STATUS.warning }}>
          {warning}
        </p>
      ))}
    </div>
  );
}

/**
 * A macro average hides which class the model is failing on. This is the row
 * that shows a minority class scoring zero while accuracy looks healthy.
 */
function PerClassTable({ rows }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div className="summary-table-block">
      <div className="chart-table-wrap">
        <table className="chart-table">
          <caption className="summary-table-caption">Per-class performance (held-out rows)</caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              <th scope="col">Precision</th>
              <th scope="col">Recall</th>
              <th scope="col">F1</th>
              <th scope="col">Support</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <th scope="row" className="summary-column-name">
                  {row.label}
                </th>
                <td className="tabular">{formatNumber(row.precision, 3)}</td>
                <td className="tabular">{formatNumber(row.recall, 3)}</td>
                <td className={row.f1 === 0 ? "tabular value-warning" : "tabular"}>{formatNumber(row.f1, 3)}</td>
                <td className="tabular">{row.support.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TuningTable({ reports }) {
  if (!reports || reports.length === 0) return null;
  return (
    <div className="summary-table-block">
      <div className="chart-table-wrap">
        <table className="chart-table">
          <caption className="summary-table-caption">
            <SlidersHorizontal size={12} style={{ marginRight: 6, verticalAlign: "-1px" }} />
            Hyperparameter search
          </caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Candidates</th>
              <th scope="col">Seconds</th>
              <th scope="col">Best CV score</th>
              <th scope="col">Chosen parameters</th>
            </tr>
          </thead>
          <tbody>
            {reports.map((report) => (
              <tr key={report.model}>
                <th scope="row" className="summary-column-name">
                  {report.model}
                </th>
                <td className="tabular">{report.candidates}</td>
                <td className="tabular">{report.seconds}</td>
                <td className="tabular">{formatNumber(report.best_cv_score, 4)}</td>
                <td>
                  <code className="params">
                    {Object.entries(report.best_params || {})
                      .map(([key, value]) => `${key}=${value}`)
                      .join(", ") || "defaults"}
                  </code>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** A library upgrade can move a score without anything in the project changing. */
function Libraries({ versions }) {
  const [open, setOpen] = useState(false);
  const entries = Object.entries(versions || {});
  if (entries.length === 0) return null;

  return (
    <div className="summary-table-block">
      <button type="button" className="chart-toggle" onClick={() => setOpen((current) => !current)} aria-expanded={open}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        Library versions
      </button>
      {open && (
        <ul className="library-list">
          {entries.map(([name, value]) => (
            <li key={name}>
              <span>{name}</span>
              <span className="tabular">{value || "not installed"}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
