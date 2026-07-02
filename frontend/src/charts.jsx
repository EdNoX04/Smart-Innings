import React from "react";
import { meta } from "./teams.js";

/* Lightweight hand-drawn SVG charts — no external chart library. */

// Line chart: batting win% vs wickets in hand (sensitivity).
export function SensitivityChart({ data, team, currentWickets }) {
  if (!data || !data.length) return null;
  const W = 360, H = 180, pad = 34;
  const xs = data.map((d) => d.wickets_in_hand);
  const maxX = 10, minX = 1;
  const px = (x) => pad + ((x - minX) / (maxX - minX)) * (W - pad - 12);
  const py = (p) => H - pad - (p / 100) * (H - pad - 14);
  const pts = data
    .filter((d) => d.wickets_in_hand >= 1)
    .map((d) => `${px(d.wickets_in_hand)},${py(d.prob)}`)
    .join(" ");
  const c = meta(team).primary;
  const curW = 10 - currentWickets;

  return (
    <div className="chart">
      <div className="chart-title">Win % vs wickets in hand</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {[0, 25, 50, 75, 100].map((g) => (
          <g key={g}>
            <line x1={pad} y1={py(g)} x2={W - 12} y2={py(g)} className="grid" />
            <text x={pad - 6} y={py(g) + 3} className="axis" textAnchor="end">{g}</text>
          </g>
        ))}
        {[2, 4, 6, 8, 10].map((x) => (
          <text key={x} x={px(x)} y={H - pad + 15} className="axis" textAnchor="middle">{x}</text>
        ))}
        <line x1={pad} y1={py(50)} x2={W - 12} y2={py(50)} className="mid-line" />
        <polyline points={pts} fill="none" stroke={c} strokeWidth="2.5" />
        {data.filter((d) => d.wickets_in_hand >= 1).map((d) => (
          <circle key={d.wickets_in_hand} cx={px(d.wickets_in_hand)} cy={py(d.prob)}
            r={d.wickets_in_hand === curW ? 5 : 3}
            fill={d.wickets_in_hand === curW ? "#fff" : c}
            stroke={c} strokeWidth={d.wickets_in_hand === curW ? 3 : 0} />
        ))}
        <text x={W - 12} y={14} className="axis" textAnchor="end">● = current</text>
      </svg>
    </div>
  );
}

// Reliability diagram: model confidence bucket vs actual held-out accuracy.
export function CalibrationChart({ calibration }) {
  if (!calibration || !calibration.bins) return null;
  const bins = calibration.bins.filter((b) => b.acc != null);
  const W = 360, H = 190, pad = 34, bw = (W - pad - 14) / bins.length;
  const py = (p) => H - pad - p * (H - pad - 16);
  return (
    <div className="chart">
      <div className="chart-title">
        Reliability — how often the model is right at each confidence
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="xMidYMid meet">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={pad} y1={py(g)} x2={W - 12} y2={py(g)} className="grid" />
            <text x={pad - 6} y={py(g) + 3} className="axis" textAnchor="end">{Math.round(g * 100)}</text>
          </g>
        ))}
        {bins.map((b, i) => {
          const x = pad + i * bw + 4;
          const h = (H - pad - 16) * b.acc;
          const good = b.acc >= 0.95;
          const midHi = b.acc >= 0.8;
          return (
            <g key={i}>
              <rect x={x} y={py(b.acc)} width={bw - 8} height={h} rx="3"
                fill={good ? "#34d399" : midHi ? "#5b8cff" : "#64748b"} opacity="0.9" />
              <text x={x + (bw - 8) / 2} y={py(b.acc) - 5} className="bar-val" textAnchor="middle">
                {Math.round(b.acc * 100)}%
              </text>
              <text x={x + (bw - 8) / 2} y={H - pad + 15} className="axis" textAnchor="middle">
                {Math.round(b.lo * 100)}-{Math.round(b.hi * 100)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="chart-foot">
        Confidence bucket (%) · measured on {calibration.n?.toLocaleString?.() || calibration.n} held-out predictions ·
        overall {Math.round(calibration.overall * 100)}%
      </div>
    </div>
  );
}
