import React from 'react';
import './RecommendationsPanel.css';

const ALERT_CONFIG = {
  'Critical Risk': { color: '#ef4444', bg: 'rgba(239,68,68,0.06)', icon: '🚨', badge: 'CRITICAL' },
  'High Risk':     { color: '#f97316', bg: 'rgba(249,115,22,0.06)', icon: '⚠️', badge: 'HIGH'     },
  'Moderate Risk': { color: '#eab308', bg: 'rgba(234,179,8,0.06)',  icon: '🟡', badge: 'MODERATE' },
  'Low Risk':      { color: '#22c55e', bg: 'rgba(34,197,94,0.06)',  icon: '✅', badge: 'LOW'       },
};

const REC_ICONS = {
  'URGENT':        '🔴',
  'Field':         '🔍',
  'Potential':     '🌲',
  'Forest':        '🌳',
  'Urban':         '🏙️',
  'Agricultural':  '🌾',
  'Barren':        '🏜️',
  'Monitor':       '📡',
  'No significant':'📊',
  'Net forest gain':'🌿',
};

function getRecIcon(text) {
  for (const [key, icon] of Object.entries(REC_ICONS)) {
    if (text.startsWith(key)) return icon;
  }
  return '📌';
}

export default function RecommendationsPanel({ result }) {
  const { recommendations, alert_level, report_url, region, year1, year2 } = result;
  const cfg = ALERT_CONFIG[alert_level] || ALERT_CONFIG['Low Risk'];

  const handleDownload = () => {
    if (report_url) window.open(`http://localhost:8001${report_url}`, '_blank');
  };

  return (
    <section className="recs-section card">
      <div className="recs-header">
        <div>
          <h2>Recommendations & Actions</h2>
          <div className="recs-context">
            Based on analysis of <strong>{region}</strong> between <strong>{year1}</strong> and <strong>{year2}</strong>
          </div>
        </div>

        {report_url && (
          <button
            id="download-report-btn"
            className="btn-primary download-btn"
            onClick={handleDownload}
          >
            <span>⬇</span> Download PDF Report
          </button>
        )}
      </div>

      {/* Alert indicator */}
      <div className="recs-alert-pill" style={{ background: cfg.bg, borderColor: cfg.color + '44' }}>
        <span className="recs-alert-icon">{cfg.icon}</span>
        <span className="recs-alert-badge" style={{ color: cfg.color }}>
          {cfg.badge} RISK
        </span>
        <span className="recs-alert-desc" style={{ color: cfg.color }}>
          — {alert_level}
        </span>
      </div>

      {/* Recommendations list */}
      <ul className="recs-list">
        {recommendations.map((rec, i) => (
          <li key={i} className="recs-item">
            <span className="recs-item-icon">{getRecIcon(rec)}</span>
            <span className="recs-item-text">{rec}</span>
          </li>
        ))}
      </ul>

      {/* Legend for change map */}
      <div className="recs-change-legend">
        <div className="section-label" style={{ marginBottom: 10 }}>Change Map Legend</div>
        <div className="change-legend-grid">
          {[
            { color: '#dc3232', label: 'Forest Loss',   desc: 'Forest → any class' },
            { color: '#32c832', label: 'Forest Gain',   desc: 'Any → Forest' },
            { color: '#ff8c00', label: 'Urbanisation',  desc: 'Any → Urban' },
            { color: '#c8c8c8', label: 'No Change',     desc: 'Same class' },
            { color: '#ffffff', label: 'Other Change',  desc: 'Non-forest transition' },
          ].map(item => (
            <div key={item.label} className="change-legend-item">
              <span className="change-legend-dot" style={{ background: item.color }} />
              <div>
                <div className="change-legend-label">{item.label}</div>
                <div className="change-legend-desc">{item.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
