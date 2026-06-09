import React, { useState } from 'react';
import './ImageGrid.css';

const VIEWS = [
  { key: 'satellite_y1',   label: 'Satellite Y1',    icon: '🛰️' },
  { key: 'satellite_y2',   label: 'Satellite Y2',    icon: '🛰️' },
  { key: 'segmentation_y1',label: 'Seg Map Y1',      icon: '🗺️' },
  { key: 'segmentation_y2',label: 'Seg Map Y2',      icon: '🗺️' },
  { key: 'change_map',     label: 'Change Map',       icon: '🔴' },
];

const CLASS_LEGEND = [
  { color: '#1a9641', label: 'Forest' },
  { color: '#ffffb2', label: 'Agriculture' },
  { color: '#d7191c', label: 'Urban' },
  { color: '#2c7bb6', label: 'Water' },
  { color: '#d9d9d9', label: 'Barren' },
];

const CHANGE_LEGEND = [
  { color: '#dc3232', label: 'Forest Loss' },
  { color: '#32c832', label: 'Forest Gain' },
  { color: '#ff8c00', label: 'Urbanisation' },
  { color: '#c8c8c8', label: 'No Change' },
  { color: '#ffffff', label: 'Other Change' },
];

function ImageTile({ title, icon, src, isActive, onClick, year }) {
  const [zoomed, setZoomed] = useState(false);
  return (
    <>
      <div
        className={`img-tile ${isActive ? 'img-tile--active' : ''}`}
        onClick={onClick}
        title={`Click to expand: ${title}`}
      >
        <div className="img-tile-header">
          <span className="img-tile-icon">{icon}</span>
          <span className="img-tile-title">{title}</span>
          {year && <span className="img-tile-year">{year}</span>}
          <button
            className="img-tile-zoom-btn"
            onClick={e => { e.stopPropagation(); setZoomed(true); }}
            title="Full screen"
          >⤢</button>
        </div>
        <div className="img-tile-body">
          {src
            ? <img src={`data:image/png;base64,${src}`} alt={title} className="img-tile-img" />
            : <div className="img-tile-placeholder">No image</div>
          }
        </div>
      </div>

      {/* Full-screen overlay */}
      {zoomed && (
        <div className="img-overlay" onClick={() => setZoomed(false)}>
          <div className="img-overlay-inner">
            <button className="img-overlay-close" onClick={() => setZoomed(false)}>✕</button>
            <div className="img-overlay-label">{icon} {title}</div>
            <img
              src={`data:image/png;base64,${src}`}
              alt={title}
              className="img-overlay-img"
              onClick={e => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </>
  );
}

export default function ImageGrid({ result }) {
  const { images, year1, year2 } = result;
  const [featured, setFeatured] = useState('satellite_y1');
  const isChange = featured === 'change_map';
  const legend   = isChange ? CHANGE_LEGEND : CLASS_LEGEND;
  const showLegend = featured.startsWith('seg') || isChange;

  const featuredView = VIEWS.find(v => v.key === featured);

  return (
    <section className="image-section card">
      <h2>Satellite Imagery & Segmentation Maps</h2>

      {/* Main featured view */}
      <div className="img-featured-wrapper">
        <div className="img-featured">
          {images[featured]
            ? <img
                src={`data:image/png;base64,${images[featured]}`}
                alt={featuredView?.label}
                className="img-featured-img"
              />
            : <div className="img-featured-placeholder">Select a view below</div>
          }
          <div className="img-featured-label">
            {featuredView?.icon} {featuredView?.label}
          </div>
        </div>

        {showLegend && (
          <div className="img-legend">
            <div className="img-legend-title">Legend</div>
            {legend.map(item => (
              <div key={item.label} className="img-legend-item">
                <span className="img-legend-dot" style={{ background: item.color }} />
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Thumbnail strip */}
      <div className="img-strip">
        {VIEWS.map(v => (
          <ImageTile
            key={v.key}
            icon={v.icon}
            title={v.label}
            src={images[v.key]}
            isActive={featured === v.key}
            onClick={() => setFeatured(v.key)}
            year={v.key.endsWith('_y1') ? year1 : v.key.endsWith('_y2') ? year2 : null}
          />
        ))}
      </div>
    </section>
  );
}
