export default function RiskGauge({ score }) {
  const normalized = Math.max(0, Math.min(100, Number(score || 0)));
  const hue = Math.max(5, 120 - normalized * 1.1);
  return (
    <section className="card">
      <h2>Risk Gauge</h2>
      <div className="gauge">
        <div className="gauge-fill" style={{ width: `${normalized}%`, background: `hsl(${hue} 80% 48%)` }} />
      </div>
      <p className="gauge-number">{normalized.toFixed(1)} / 100</p>
    </section>
  );
}
