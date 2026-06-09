import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { api } from './api';
import ControlPanel from './components/ControlPanel';
import DashboardCards from './components/DashboardCards';
import ImageGrid from './components/ImageGrid';
import ChartsSection from './components/ChartsSection';
import ChangeTable from './components/ChangeTable';
import RecommendationsPanel from './components/RecommendationsPanel';
import AcademicAnalysis from './components/AcademicAnalysis';
import './App.css';

export default function App() {
  const [regions, setRegions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    api.regions()
      .then(r => setRegions(r.data.regions))
      .catch(() => toast.error('Cannot reach backend — is the server running on port 8001?'));
  }, []);

  const handleAnalyze = async ({ region, year1, year2 }) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { data } = await api.analyze(region, year1, year2);
      setResult(data);
      toast.success(`Analysis complete for ${region}!`, { icon: '🌿' });
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      setError(msg);
      toast.error(`Analysis failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-logo">
          <div className="header-logo-icon">🌿</div>
          <div className="header-title-group">
            <h1>Western Ghats Monitor</h1>
            <p>SENTINEL-2 · AI LAND-COVER SEGMENTATION · CHANGE DETECTION</p>
          </div>
        </div>

        <div className="header-spacer" />

        <div className="header-status">
          <span className="status-dot" />
          <span>System Active</span>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="app-main">
        <ControlPanel
          regions={regions}
          onAnalyze={handleAnalyze}
          loading={loading}
        />

        {loading && (
          <div className="loading-banner">
            <div className="spinner" />
            <div>
              <div>Fetching Sentinel-2 imagery and running AI segmentation — this may take 1–3 minutes…</div>
              <div className="loading-steps">
                <span className="loading-step active">⟳ Fetching satellite data</span>
                <span className="loading-step">⟳ Running U-Net segmentation</span>
                <span className="loading-step">⟳ Computing change detection</span>
                <span className="loading-step">⟳ Generating report</span>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            <div>
              <strong>Analysis failed</strong>
              <div style={{ marginTop: 4, opacity: 0.8 }}>{error}</div>
            </div>
          </div>
        )}

        {result && (
          <div className="results-wrapper">
            <DashboardCards result={result} />
            <ImageGrid result={result} />
            <ChartsSection result={result} />
            <ChangeTable result={result} />
            <RecommendationsPanel result={result} />
            <AcademicAnalysis result={result} />
          </div>
        )}
      </main>

      <footer className="app-footer">
        Western Ghats Deforestation Monitor · Sentinel-2 + U-Net AI · Data source: Google Earth Engine
      </footer>

      <ToastContainer
        theme="dark"
        position="bottom-right"
        autoClose={4000}
        toastStyle={{ background: '#111a14', border: '1px solid rgba(74,222,128,0.15)', color: '#e8f5e9' }}
      />
    </div>
  );
}
