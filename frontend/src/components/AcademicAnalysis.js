import React from 'react';
import './AcademicAnalysis.css';

export default function AcademicAnalysis({ result }) {
  const { academic_analysis, region, year1, year2 } = result;

  if (!academic_analysis) return null;

  return (
    <section className="academic-section card">
      <div className="academic-header">
        <h2>Academic Analysis & Ecological Justification</h2>
        <div className="academic-context">
          Insights based on land-cover transitions in <strong>{region}</strong> ({year1} - {year2})
        </div>
      </div>

      <div className="academic-content">
        <div className="analysis-block">
          <h3>1. Reasoning for Land Cover Shifts</h3>
          <p>{academic_analysis.reasoning}</p>
        </div>

        <div className="analysis-block">
          <h3>2. Steps for Ecological Balance</h3>
          <ul className="steps-list">
            {academic_analysis.steps.map((step, i) => {
              const [boldPart, restPart] = step.split('**').filter(Boolean);
              return (
                <li key={i}>
                  <strong>{boldPart}</strong>
                  {restPart}
                </li>
              );
            })}
          </ul>
        </div>

        <div className="analysis-block">
          <h3>3. Conclusive Points</h3>
          <ul className="conclusions-list">
            {academic_analysis.conclusions.slice(0, 3).map((conc, i) => {
              const [boldPart, restPart] = conc.split('**').filter(Boolean);
              return (
                <li key={i}>
                  <strong>{boldPart}</strong>
                  {restPart}
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </section>
  );
}
