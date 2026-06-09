import React from 'react';
import './ChangeTable.css';

const CLASS_COLORS = {
  Forest:      '#22c55e',
  Agriculture: '#eab308',
  Urban:       '#f97316',
  Water:       '#3b82f6',
  Barren:      '#94a3b8',
};

function ChangeCell({ value }) {
  const num  = parseFloat(value);
  const sign = num > 0 ? '+' : '';
  const cls  = num > 0.5 ? 'change-positive'
             : num < -0.5 ? 'change-negative'
             : 'change-neutral';
  return (
    <td className={`change-cell ${cls}`}>
      {sign}{num.toFixed(2)}%
    </td>
  );
}

export default function ChangeTable({ result }) {
  const { class_pct_y1, class_pct_y2, class_changes, transitions, forest, year1, year2 } = result;
  const classes = Object.keys(class_pct_y1);

  return (
    <section className="change-table-section card">
      <h2>Detailed Land Cover Statistics</h2>

      {/* Land cover table */}
      <div className="table-wrapper">
        <table className="data-table" id="land-cover-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>{year1} Coverage</th>
              <th>{year2} Coverage</th>
              <th>Δ Change</th>
              <th>Trend</th>
            </tr>
          </thead>
          <tbody>
            {classes.map(name => {
              const p1  = class_pct_y1[name] ?? 0;
              const p2  = class_pct_y2[name] ?? 0;
              const chg = class_changes[name]  ?? 0;
              return (
                <tr key={name}>
                  <td>
                    <span
                      className="class-dot"
                      style={{ background: CLASS_COLORS[name] || '#ccc' }}
                    />
                    {name}
                  </td>
                  <td>{p1.toFixed(2)}%</td>
                  <td>{p2.toFixed(2)}%</td>
                  <ChangeCell value={chg} />
                  <td>
                    <div className="sparkbar-track">
                      <div
                        className="sparkbar-fill"
                        style={{
                          width: `${Math.min(p2, 100)}%`,
                          background: CLASS_COLORS[name],
                        }}
                      />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="divider" />

      {/* Forest summary row */}
      <div className="forest-summary">
        <div className="forest-summary-item">
          <span className="fs-label">Forest {year1}</span>
          <span className="fs-value" style={{ color: '#22c55e' }}>{forest.year1_pct.toFixed(2)}%</span>
        </div>
        <div className="fs-arrow">→</div>
        <div className="forest-summary-item">
          <span className="fs-label">Forest {year2}</span>
          <span className="fs-value" style={{ color: forest.year2_pct < forest.year1_pct ? '#ef4444' : '#22c55e' }}>
            {forest.year2_pct.toFixed(2)}%
          </span>
        </div>
        <div className="fs-divider" />
        <div className="forest-summary-item">
          <span className="fs-label">Net Change</span>
          <span className="fs-value" style={{ color: forest.net_change_pct >= 0 ? '#22c55e' : '#ef4444' }}>
            {forest.net_change_pct > 0 ? '+' : ''}{forest.net_change_pct.toFixed(2)}%
          </span>
        </div>
      </div>

      <div className="divider" />

      {/* Transition table */}
      <div className="section-label" style={{ marginTop: 4 }}>Transition Matrix</div>
      <div className="table-wrapper">
        <table className="data-table" id="transition-table">
          <thead>
            <tr>
              <th>Transition</th>
              <th>Area (%)</th>
              <th>Severity</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(transitions).map(([key, val]) => {
              const sev = val > 2 ? 'High' : val > 0.5 ? 'Moderate' : 'Low';
              const sevColor = sev === 'High' ? '#ef4444' : sev === 'Moderate' ? '#eab308' : '#22c55e';
              return (
                <tr key={key}>
                  <td className="transition-key">{key}</td>
                  <td>{val.toFixed(3)}%</td>
                  <td>
                    <span className="sev-badge" style={{ color: sevColor, borderColor: sevColor + '44' }}>
                      {sev}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
