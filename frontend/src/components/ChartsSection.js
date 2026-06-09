import React from 'react';
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import './ChartsSection.css';

const CLASS_COLORS = {
  Forest:      '#22c55e',
  Agriculture: '#eab308',
  Urban:       '#f97316',
  Water:       '#3b82f6',
  Barren:      '#94a3b8',
};

const TOOLTIP_STYLE = {
  background: '#111a14',
  border: '1px solid rgba(74,222,128,0.15)',
  borderRadius: '10px',
  color: '#e8f5e9',
  fontSize: '0.8rem',
  padding: '10px 14px',
};

function PctTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={TOOLTIP_STYLE}>
      <strong>{payload[0].name}</strong>
      <div>{payload[0].value.toFixed(2)}%</div>
    </div>
  );
}

function ChangeTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={TOOLTIP_STYLE}>
      <strong>{label}</strong>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(2)}%
        </div>
      ))}
    </div>
  );
}

function renderCustomLabel({ name, percent }) {
  if (percent < 0.05) return null;
  return `${(percent * 100).toFixed(1)}%`;
}

export default function ChartsSection({ result }) {
  const { class_pct_y1, class_pct_y2, class_changes, year1, year2, transitions } = result;

  // Data for pie charts
  const pieData1 = Object.entries(class_pct_y1).map(([name, value]) => ({ name, value }));
  const pieData2 = Object.entries(class_pct_y2).map(([name, value]) => ({ name, value }));

  // Data for comparison bar chart
  const barData = Object.keys(class_pct_y1).map(name => ({
    name,
    [year1]: parseFloat(class_pct_y1[name].toFixed(2)),
    [year2]: parseFloat(class_pct_y2[name].toFixed(2)),
  }));

  // Change bar chart
  const changeData = Object.entries(class_changes).map(([name, value]) => ({
    name,
    change: parseFloat(value.toFixed(3)),
  }));

  // Transitions bar chart
  const transData = Object.entries(transitions).map(([name, value]) => ({
    name: name.replace(' → ', ' →\n'),
    pct: parseFloat(value.toFixed(3)),
  }));

  return (
    <section className="charts-section">
      <div className="charts-grid">

        {/* Pie Y1 */}
        <div className="chart-card card">
          <div className="chart-title">Land Cover — {year1}</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData1}
                dataKey="value"
                cx="50%" cy="50%"
                innerRadius={52}
                outerRadius={88}
                paddingAngle={2}
                label={renderCustomLabel}
                labelLine={false}
              >
                {pieData1.map(entry => (
                  <Cell key={entry.name} fill={CLASS_COLORS[entry.name]} />
                ))}
              </Pie>
              <Tooltip content={<PctTooltip />} />
              <Legend
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: '0.72rem', color: '#86a88e' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Pie Y2 */}
        <div className="chart-card card">
          <div className="chart-title">Land Cover — {year2}</div>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData2}
                dataKey="value"
                cx="50%" cy="50%"
                innerRadius={52}
                outerRadius={88}
                paddingAngle={2}
                label={renderCustomLabel}
                labelLine={false}
              >
                {pieData2.map(entry => (
                  <Cell key={entry.name} fill={CLASS_COLORS[entry.name]} />
                ))}
              </Pie>
              <Tooltip content={<PctTooltip />} />
              <Legend
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: '0.72rem', color: '#86a88e' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Grouped bar: Y1 vs Y2 */}
        <div className="chart-card chart-card--wide card">
          <div className="chart-title">Year-on-Year Comparison ({year1} vs {year2})</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} barCategoryGap="30%" barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: '#4a6352', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#4a6352', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={32}
                tickFormatter={v => `${v}%`}
              />
              <Tooltip content={<ChangeTooltip />} />
              <Legend
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: '0.72rem', color: '#86a88e' }}
              />
              <Bar dataKey={year1} fill="#4a6352" radius={[3, 3, 0, 0]} />
              <Bar dataKey={year2} fill="#22c55e" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Change bar */}
        <div className="chart-card card">
          <div className="chart-title">Class Change (Δ%)</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={changeData} layout="vertical" barCategoryGap="25%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#4a6352', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => `${v}%`}
              />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: '#86a88e', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                width={80}
              />
              <Tooltip content={<ChangeTooltip />} />
              <Bar dataKey="change" radius={[0, 3, 3, 0]}>
                {changeData.map(entry => (
                  <Cell
                    key={entry.name}
                    fill={entry.change >= 0 ? CLASS_COLORS[entry.name] || '#22c55e' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Transitions bar */}
        <div className="chart-card card">
          <div className="chart-title">Key Transitions (% of Area)</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={transData} layout="vertical" barCategoryGap="25%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#4a6352', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => `${v}%`}
              />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: '#86a88e', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={100}
              />
              <Tooltip content={<ChangeTooltip />} />
              <Bar dataKey="pct" fill="#ef4444" radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>
    </section>
  );
}
