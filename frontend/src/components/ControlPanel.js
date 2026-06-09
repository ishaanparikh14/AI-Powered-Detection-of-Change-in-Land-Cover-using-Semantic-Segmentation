import React, { useState } from 'react';
import './ControlPanel.css';

const YEARS = Array.from({ length: 10 }, (_, i) => 2015 + i);

export default function ControlPanel({ regions, onAnalyze, loading }) {
  const [region, setRegion] = useState('');
  const [year1,  setYear1]  = useState(2020);
  const [year2,  setYear2]  = useState(2024);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!region)       return alert('Please select a region.');
    if (year1 >= year2) return alert('Year 1 must be earlier than Year 2.');
    onAnalyze({ region, year1, year2 });
  };

  return (
    <section className="control-panel-wrapper">
      <div className="control-panel-header">
        <span className="control-panel-eyebrow">Analysis Configuration</span>
        <h2 className="control-panel-title">Select Region & Time Period</h2>
        <p className="control-panel-desc">
          Choose one of six Western Ghats biodiversity hotspots and a comparison window.
          Sentinel-2 imagery will be fetched automatically.
        </p>
      </div>

      <form className="control-panel-form card" onSubmit={handleSubmit}>
        {/* Region */}
        <div className="control-field">
          <label className="control-label" htmlFor="region-select">
            <span className="control-label-icon">📍</span> Region
          </label>
          <select
            id="region-select"
            className="field-select"
            value={region}
            onChange={e => setRegion(e.target.value)}
            required
          >
            <option value="">— Select a region —</option>
            {regions.map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>

        {/* Year 1 */}
        <div className="control-field">
          <label className="control-label" htmlFor="year1-select">
            <span className="control-label-icon">📅</span> Year 1 (Baseline)
          </label>
          <select
            id="year1-select"
            className="field-select"
            value={year1}
            onChange={e => setYear1(+e.target.value)}
          >
            {YEARS.map(y => <option key={y} value={y}>{y + 2}</option>)}
          </select>
        </div>

        {/* Year 2 */}
        <div className="control-field">
          <label className="control-label" htmlFor="year2-select">
            <span className="control-label-icon">📅</span> Year 2 (Comparison)
          </label>
          <select
            id="year2-select"
            className="field-select"
            value={year2}
            onChange={e => setYear2(+e.target.value)}
          >
            {YEARS.map(y => <option key={y} value={y}>{y + 2}</option>)}
          </select>
        </div>

        {/* Submit */}
        <div className="control-submit">
          <button type="submit" id="analyze-btn" className="btn-primary" disabled={loading}>
            {loading
              ? <><div className="spinner" /> Analyzing…</>
              : <><span>🔍</span> Run Analysis</>
            }
          </button>
          {region && (
            <span className="control-summary">
              {region} · {year1 + 2} → {year2 + 2}
            </span>
          )}
        </div>
      </form>
    </section>
  );
}
