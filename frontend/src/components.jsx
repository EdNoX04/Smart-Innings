import React from "react";
import { meta } from "./teams.js";

export function TeamBadge({ team, size = 34 }) {
  const m = meta(team);
  return (
    <span
      className="badge"
      style={{ background: m.primary, color: m.text, width: size, height: size, fontSize: size * 0.32 }}
      title={team}
    >
      {m.code}
    </span>
  );
}

export function Field({ label, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

export function TeamSelect({ label, value, onChange, teams, exclude }) {
  return (
    <Field label={label}>
      <div className="team-select">
        <TeamBadge team={value} size={30} />
        <select value={value} onChange={(e) => onChange(e.target.value)}>
          {teams.map((t) => (
            <option key={t} value={t} disabled={t === exclude}>
              {t}
            </option>
          ))}
        </select>
      </div>
    </Field>
  );
}

export function VenueSelect({ value, onChange, venues }) {
  return (
    <Field label="Venue">
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {venues.map((v) => (
          <option key={v} value={v}>{v}</option>
        ))}
      </select>
    </Field>
  );
}

export function NumberField({ label, value, onChange, min = 0, max = 999, suffix }) {
  return (
    <Field label={label}>
      <div className="num-wrap">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          onChange={(e) => {
            const v = e.target.value === "" ? "" : Number(e.target.value);
            onChange(v);
          }}
        />
        {suffix && <span className="suffix">{suffix}</span>}
      </div>
    </Field>
  );
}

// Two-team head-to-head probability bar
export function ProbBar({ leftTeam, rightTeam, leftPct }) {
  const lm = meta(leftTeam), rm = meta(rightTeam);
  return (
    <div className="probbar-wrap">
      <div className="probbar">
        <div className="seg left" style={{ width: `${leftPct}%`, background: lm.primary, color: lm.text }}>
          {leftPct >= 12 && <span>{leftPct}%</span>}
        </div>
        <div className="seg right" style={{ width: `${100 - leftPct}%`, background: rm.primary, color: rm.text }}>
          {100 - leftPct >= 12 && <span>{(100 - leftPct).toFixed(1)}%</span>}
        </div>
      </div>
      <div className="probbar-legend">
        <span><TeamBadge team={leftTeam} size={22} /> {meta(leftTeam).code} {leftPct}%</span>
        <span>{rm.code} {(100 - leftPct).toFixed(1)}% <TeamBadge team={rightTeam} size={22} /></span>
      </div>
    </div>
  );
}

// Circular gauge for a single probability
export function Gauge({ pct, team, label }) {
  const m = meta(team);
  const r = 80, c = 2 * Math.PI * r;
  const off = c * (1 - pct / 100);
  return (
    <div className="gauge">
      <svg viewBox="0 0 200 200" width="190" height="190">
        <circle cx="100" cy="100" r={r} className="gauge-track" />
        <circle
          cx="100" cy="100" r={r}
          className="gauge-val"
          stroke={m.primary}
          strokeDasharray={c}
          strokeDashoffset={off}
          transform="rotate(-90 100 100)"
        />
        <text x="100" y="94" className="gauge-num">{pct}%</text>
        <text x="100" y="124" className="gauge-sub">{label}</text>
      </svg>
    </div>
  );
}

export function Stat({ label, value, accent }) {
  return (
    <div className="stat">
      <div className="stat-value" style={accent ? { color: accent } : undefined}>{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export function Tabs({ tabs, active, onChange }) {
  return (
    <div className="tabs">
      {tabs.map((t) => (
        <button
          key={t.id}
          className={`tab ${active === t.id ? "active" : ""}`}
          onClick={() => onChange(t.id)}
        >
          <span className="tab-icon">{t.icon}</span>
          {t.label}
        </button>
      ))}
    </div>
  );
}
