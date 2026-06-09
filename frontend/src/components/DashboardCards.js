import React from 'react';
import './DashboardCards.css';

const ALERT_CONFIG = {
  'Critical Risk': { color: '#ef4444', bg: 'rgba(239,68,68,0.08)',  icon: '🚨', border: 'rgba(239,68,68,0.3)'  },
  'High Risk':     { color: '#f97316', bg: 'rgba(249,115,22,0.08)', icon: '⚠️', border: 'rgba(249,115,22,0.3)' },
  'Moderate Risk': { color: '#eab308', bg: 'rgba(234,179,8,0.08)',  icon: '🟡', border: 'rgba(234,179,8,0.3)'  },
  'Low Risk':      { color: '#22c55e', bg: 'rgba(34,197,94,0.08)',  icon: '✅', border: 'rgba(34,197,94,0.3)'  },
};

function MetricCard({ label, value, unit = '%', color, trend, sub }) {
  const numVal = typeof value === 'number' ? value : parseFloat(value);
  const display = isNaN(numVal) ? '—' : (Math.abs(numVal) < 10 ? numVal.toFixed(2) : numVal.toFixed(1));
  const sign    = numVal > 0 ? '+' : '';

  return (
    <div className="metric-card" style={{ '--accent': color }}>
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        {trend && <span className="metric-trend" style={{ color }}>{trend}</span>}
      </div>
      <div className="metric-value" style={{ color }}>
        {sign}{display}
        <span className="metric-unit">{unit}</span>
      </div>
      {sub && <div className="metric-sub">{sub}</div>}
      <div className="metric-bar-track">
        <div
          className="metric-bar-fill"
          style={{
            width: `${Math.min(Math.abs(numVal), 100)}%`,
            background: color,
          }}
        />
      </div>
    </div>
  );
}

export default function DashboardCards({ result }) {
  const { forest, class_changes, alert_level, region, year1, year2 } = result;
  const cfg = ALERT_CONFIG[alert_level] || ALERT_CONFIG['Low Risk'];

  return (
    <section className="dashboard-section">
      {/* Alert Banner */}
      <div
        className="alert-banner"
        style={{ background: cfg.bg, borderColor: cfg.border }}
      >
        <span className="alert-icon">{cfg.icon}</span>
        <div className="alert-body">
          <span className="alert-level" style={{ color: cfg.color }}>
            {alert_level}
          </span>
          <span className="alert-detail">
            Forest loss: <strong>{forest.loss_pct.toFixed(2)}%</strong> ·{' '}
            Net change: <strong>{forest.net_change_pct > 0 ? '+' : ''}{forest.net_change_pct.toFixed(2)}%</strong>
          </span>
        </div>
        <div className="alert-region">
          <span className="alert-region-name">{region}</span>
          <span className="alert-region-years">{year1} → {year2}</span>
        </div>
      </div>

      {/* Metric cards grid */}
      <div className="metrics-grid">
        <MetricCard
          label="Forest Loss"
          value={forest.loss_pct}
          color="#ef4444"
          trend="▼"
          sub={`${year1}: ${forest.year1_pct.toFixed(1)}% → ${year2}: ${forest.year2_pct.toFixed(1)}%`}
        />
        <MetricCard
          label="Forest Gain"
          value={forest.gain_pct}
          color="#22c55e"
          trend="▲"
        />
        <MetricCard
          label="Net Forest Change"
          value={forest.net_change_pct}
          color={forest.net_change_pct >= 0 ? '#22c55e' : '#ef4444'}
          trend={forest.net_change_pct >= 0 ? '▲' : '▼'}
        />
        <MetricCard
          label="Urban Growth"
          value={class_changes['Urban'] ?? 0}
          color="#f97316"
          trend={class_changes['Urban'] > 0 ? '▲' : '—'}
        />
        <MetricCard
          label="Agriculture Change"
          value={class_changes['Agriculture'] ?? 0}
          color="#eab308"
        />
        <MetricCard
          label="Barren Land Change"
          value={class_changes['Barren'] ?? 0}
          color="#94a3b8"
        />
      </div>
    </section>
  );
}
