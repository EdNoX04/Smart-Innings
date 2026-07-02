import React, { useEffect, useMemo, useState, useCallback } from "react";
import {
  getMeta, predictChase, predictChaseSweep, predictScore, predictPrematch,
  startRefresh, refreshStatus,
} from "./api.js";
import {
  Tabs, TeamSelect, VenueSelect, NumberField, Field,
  ProbBar, Gauge, Stat, TeamBadge,
} from "./components.jsx";
import { SensitivityChart, CalibrationChart } from "./charts.jsx";

// coerce a possibly-empty numeric field to a safe number
const N = (v, d = 0) => (v === "" || v == null || Number.isNaN(Number(v)) ? d : Number(v));

const TABS = [
  { id: "chase", label: "Live Win Probability", icon: "🎯" },
  { id: "score", label: "Score Projection", icon: "📈" },
  { id: "prematch", label: "Pre-Match", icon: "🪙" },
];

function useDebounced(fn, delay, deps) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const cb = useCallback(fn, deps);
  useEffect(() => {
    const t = setTimeout(cb, delay);
    return () => clearTimeout(t);
  }, [cb, delay]);
}

export default function App() {
  const [m, setM] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("chase");

  useEffect(() => {
    getMeta().then(setM).catch((e) => setErr(e.message));
  }, []);

  const reloadMeta = useCallback(() => getMeta().then(setM).catch(() => {}), []);

  if (err) return <Shell><ApiError msg={err} /></Shell>;
  if (!m) return <Shell><div className="loading">Loading models…</div></Shell>;

  return (
    <Shell meta={m} onUpdated={reloadMeta}>
      <Tabs tabs={TABS} active={tab} onChange={setTab} />
      {tab === "chase" && <ChasePanel m={m} />}
      {tab === "score" && <ScorePanel m={m} />}
      {tab === "prematch" && <PreMatchPanel m={m} />}
    </Shell>
  );
}

function Shell({ children, meta, onUpdated }) {
  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="logo">🏏</span>
          <div>
            <h1>SmartInnings</h1>
            <p>IPL match predictor · trained on 2008–2026 ball-by-ball data</p>
          </div>
        </div>
        {meta && (
          <div className="header-right">
            <div className="accuracy-chips">
              <span className="chip">Chase acc {(meta.models.chase.test_acc * 100).toFixed(0)}%</span>
              <span className="chip">Chase AUC {meta.models.chase.test_auc.toFixed(2)}</span>
              <span className="chip">Score MAE ±{Math.round(meta.models.score.test_mae)}</span>
            </div>
            <RefreshControl onUpdated={onUpdated} enabled={meta.update_enabled} />
          </div>
        )}
      </header>
      <main>{children}</main>
      <footer className="footer">
        Models served from a from-scratch numpy ML stack (gradient-boosted trees · logistic regression).
        Predictions are estimates for entertainment and analysis.
      </footer>
    </div>
  );
}

function RefreshControl({ onUpdated, enabled }) {
  // On a cloud deploy where the update button isn't configured, show an
  // informational chip instead of a broken button. (Kept above hooks so hook
  // order stays consistent.)
  if (enabled === false) {
    return <span className="chip" title="Data refresh + retrain runs automatically via GitHub Actions">🔄 Auto-updates weekly</span>;
  }
  return <RefreshButton onUpdated={onUpdated} />;
}

function RefreshButton({ onUpdated }) {
  const [st, setSt] = useState(null);     // latest status from backend
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);

  // poll while a refresh is running
  useEffect(() => {
    if (!busy) return;
    let alive = true;
    const tick = async () => {
      try {
        const s = await refreshStatus();
        if (!alive) return;
        setSt(s);
        if (!s.running) {
          setBusy(false);
          if (s.step === "done") onUpdated && onUpdated();
        }
      } catch {
        if (alive) setBusy(false);
      }
    };
    const id = setInterval(tick, 1500);
    tick();
    return () => { alive = false; clearInterval(id); };
  }, [busy, onUpdated]);

  const onClick = async () => {
    setOpen(true);
    try {
      const s = await startRefresh();
      setSt(s);
      setBusy(true);
    } catch (e) {
      setSt({ step: "error", message: e.message });
    }
  };

  const label = busy
    ? (st?.step === "training" ? "Retraining…" : st?.step === "reloading" ? "Reloading…" : "Updating…")
    : "↻ Update data";

  return (
    <div className="refresh">
      <button className="btn refresh-btn" onClick={onClick} disabled={busy}>
        {busy && <span className="spinner" />}
        {label}
      </button>
      {open && st && (
        <div className={`refresh-status ${st.step}`}>
          <span>{st.message}</span>
          {st.step === "done" && st.last_updated && (
            <span className="muted-sm"> ({new Date(st.last_updated).toLocaleString()})</span>
          )}
          {!busy && (
            <button className="x" onClick={() => setOpen(false)} aria-label="dismiss">×</button>
          )}
        </div>
      )}
    </div>
  );
}

function ApiError({ msg }) {
  return (
    <div className="api-error">
      <h2>⚠️ Can't reach the SmartInnings API</h2>
      <p>{msg}</p>
      <p>Start the backend, then refresh:</p>
      <pre>{`cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000`}</pre>
    </div>
  );
}

/* --------------------------------------------------------------------- */
/*  Chase win probability                                                */
/* --------------------------------------------------------------------- */
function ChasePanel({ m }) {
  const teams = m.teams;
  const [bat, setBat] = useState(teams[0]);
  const [bowl, setBowl] = useState(teams[1]);
  const [venue, setVenue] = useState(m.venues[0]);
  const [target, setTarget] = useState(180);
  const [score, setScore] = useState(90);
  const [wickets, setWickets] = useState(2);
  const [overs, setOvers] = useState(12);
  const [balls, setBalls] = useState(0);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState([]);
  const [sweep, setSweep] = useState(null);

  const valid = bat !== bowl && N(target) > 0;

  useDebounced(() => {
    if (!valid) return;
    const body = {
      batting_team: bat, bowling_team: bowl, venue,
      target: N(target, 1), score: N(score), wickets: N(wickets),
      overs: N(overs), balls: N(balls),
    };
    setBusy(true);
    predictChase(body)
      .then((r) => { setRes(r); setBusy(false); })
      .catch(() => setBusy(false));
    predictChaseSweep(body).then(setSweep).catch(() => setSweep(null));
  }, 250, [bat, bowl, venue, target, score, wickets, overs, balls]);

  const snapshot = () => {
    if (!res) return;
    setHistory((h) => [{
      id: Date.now(), bat, bowl,
      desc: `${res.runs_left} off ${res.balls_left}`,
      prob: res.batting_win_prob,
    }, ...h].slice(0, 5));
  };

  return (
    <div className="panel">
      <section className="inputs card">
        <h3>Second innings — match situation</h3>
        <div className="grid2">
          <TeamSelect label="Batting (chasing)" value={bat} onChange={setBat} teams={teams} exclude={bowl} />
          <TeamSelect label="Bowling (defending)" value={bowl} onChange={setBowl} teams={teams} exclude={bat} />
        </div>
        <VenueSelect value={venue} onChange={setVenue} venues={m.venues} />
        <div className="grid3">
          <NumberField label="Target" value={target} onChange={setTarget} min={1} max={350} />
          <NumberField label="Current score" value={score} onChange={setScore} min={0} max={350} />
          <NumberField label="Wickets lost" value={wickets} onChange={setWickets} min={0} max={9} />
        </div>
        <div className="grid3">
          <NumberField label="Overs" value={overs} onChange={setOvers} min={0} max={19} />
          <NumberField label="Balls" value={balls} onChange={setBalls} min={0} max={5} />
          <div className="field">
            <span className="field-label">&nbsp;</span>
            <button className="btn" disabled={!res} onClick={snapshot}>＋ Save scenario</button>
          </div>
        </div>
        {!valid && <p className="warn">Batting and bowling teams must differ.</p>}
      </section>

      <section className="output card">
        {res ? (
          <>
            <ProbBar leftTeam={bat} rightTeam={bowl} leftPct={res.batting_win_prob} />
            <div className="badge-row">
              <MetricBadge
                label="Clarity"
                pct={res.confidence}
                caption="how decisive the situation is"
                tone={res.confidence >= 90 ? "high" : res.confidence >= 70 ? "mid" : "low"}
              />
              {res.situation_accuracy != null && (
                <MetricBadge
                  label="Accuracy"
                  pct={res.situation_accuracy}
                  caption={`right this often when this sure${res.situation_n ? ` · n=${res.situation_n}` : ""}`}
                  tone={res.situation_accuracy >= 95 ? "high" : res.situation_accuracy >= 80 ? "mid" : "low"}
                />
              )}
            </div>
            <div className="stats-row">
              <Stat label="Runs needed" value={res.runs_left} />
              <Stat label="Balls left" value={res.balls_left} />
              <Stat label="Req. RR" value={res.rrr} accent="#fb7185" />
              <Stat label="Current RR" value={res.crr} accent="#34d399" />
            </div>
            {sweep && <SensitivityChart data={sweep.by_wickets} team={bat} currentWickets={N(wickets)} />}
            <p className="hint">{busy ? "updating…" : "Updates live as you change the situation."}</p>
          </>
        ) : <div className="loading">Enter a match situation…</div>}

        {history.length > 0 && (
          <div className="history">
            <h4>Saved scenarios</h4>
            {history.map((h) => (
              <div className="history-row" key={h.id}>
                <TeamBadge team={h.bat} size={22} />
                <span className="h-desc">{h.desc}</span>
                <div className="h-bar"><div style={{ width: `${h.prob}%` }} /></div>
                <span className="h-prob">{h.prob}%</span>
              </div>
            ))}
          </div>
        )}
      </section>

      {m.calibration && (
        <section className="card span2">
          <CalibrationChart calibration={m.calibration} />
        </section>
      )}
    </div>
  );
}

function MetricBadge({ label, pct, caption, tone }) {
  return (
    <div className={`metric-badge ${tone}`}>
      <div className="mb-top">
        <span className="mb-label">{label}</span>
        <span className="mb-pct">{pct}%</span>
      </div>
      <div className="mb-track"><div className="mb-fill" style={{ width: `${pct}%` }} /></div>
      <div className="mb-cap">{caption}</div>
    </div>
  );
}

/* --------------------------------------------------------------------- */
/*  First-innings score projection                                       */
/* --------------------------------------------------------------------- */
function ScorePanel({ m }) {
  const teams = m.teams;
  const [bat, setBat] = useState(teams[0]);
  const [bowl, setBowl] = useState(teams[1]);
  const [venue, setVenue] = useState(m.venues[0]);
  const [score, setScore] = useState(60);
  const [wickets, setWickets] = useState(1);
  const [overs, setOvers] = useState(8);
  const [balls, setBalls] = useState(0);
  const [res, setRes] = useState(null);
  const valid = bat !== bowl;

  useDebounced(() => {
    if (!valid || score === "") return;
    predictScore({ batting_team: bat, bowling_team: bowl, venue, score, wickets, overs, balls })
      .then(setRes).catch(() => {});
  }, 250, [bat, bowl, venue, score, wickets, overs, balls]);

  const pct = res ? Math.min(100, Math.round((score / res.projected_score) * 100)) : 0;

  return (
    <div className="panel">
      <section className="inputs card">
        <h3>First innings — project the final total</h3>
        <div className="grid2">
          <TeamSelect label="Batting" value={bat} onChange={setBat} teams={teams} exclude={bowl} />
          <TeamSelect label="Bowling" value={bowl} onChange={setBowl} teams={teams} exclude={bat} />
        </div>
        <VenueSelect value={venue} onChange={setVenue} venues={m.venues} />
        <div className="grid2">
          <NumberField label="Current score" value={score} onChange={setScore} min={0} max={350} />
          <NumberField label="Wickets lost" value={wickets} onChange={setWickets} min={0} max={9} />
        </div>
        <div className="grid2">
          <NumberField label="Overs" value={overs} onChange={setOvers} min={0} max={19} />
          <NumberField label="Balls" value={balls} onChange={setBalls} min={0} max={5} />
        </div>
      </section>

      <section className="output card center">
        {res ? (
          <>
            <div className="proj">
              <div className="proj-num">{res.projected_score}</div>
              <div className="proj-label">projected final score</div>
              <div className="proj-range">likely range {res.low}–{res.high}</div>
            </div>
            <div className="run-track">
              <div className="run-fill" style={{ width: `${pct}%` }} />
              <span className="run-now">{score}</span>
            </div>
            <div className="stats-row">
              <Stat label="Current RR" value={res.current_run_rate} accent="#34d399" />
              <Stat label="Overs done" value={`${overs}.${balls}`} />
              <Stat label="Proj. RR" value={(res.projected_score / 20).toFixed(2)} />
            </div>
          </>
        ) : <div className="loading">Enter the first-innings state…</div>}
      </section>
    </div>
  );
}

/* --------------------------------------------------------------------- */
/*  Pre-match prediction                                                 */
/* --------------------------------------------------------------------- */
function PreMatchPanel({ m }) {
  const teams = m.teams;
  const [a, setA] = useState(teams[0]);
  const [b, setB] = useState(teams[1]);
  const [venue, setVenue] = useState(m.venues[0]);
  const [tossWinner, setTossWinner] = useState(teams[0]);
  const [tossDec, setTossDec] = useState("bat");
  const [res, setRes] = useState(null);
  const valid = a !== b;

  useEffect(() => { if (tossWinner !== a && tossWinner !== b) setTossWinner(a); }, [a, b, tossWinner]);

  useDebounced(() => {
    if (!valid) return;
    predictPrematch({ team_a: a, team_b: b, venue, toss_winner: tossWinner, toss_decision: tossDec })
      .then(setRes).catch(() => {});
  }, 200, [a, b, venue, tossWinner, tossDec]);

  return (
    <div className="panel">
      <section className="inputs card">
        <h3>Before the first ball</h3>
        <div className="grid2">
          <TeamSelect label="Team A" value={a} onChange={setA} teams={teams} exclude={b} />
          <TeamSelect label="Team B" value={b} onChange={setB} teams={teams} exclude={a} />
        </div>
        <VenueSelect value={venue} onChange={setVenue} venues={m.venues} />
        <div className="grid2">
          <Field label="Toss winner">
            <select value={tossWinner} onChange={(e) => setTossWinner(e.target.value)}>
              <option value={a}>{a}</option>
              <option value={b}>{b}</option>
            </select>
          </Field>
          <Field label="Toss decision">
            <select value={tossDec} onChange={(e) => setTossDec(e.target.value)}>
              <option value="bat">Bat first</option>
              <option value="field">Bowl first</option>
            </select>
          </Field>
        </div>
        <p className="note">
          IPL is a very high-parity league — pre-match outcomes are close to a coin flip
          (model test accuracy ≈ {(m.models.prematch.test_acc * 100).toFixed(0)}%). Treat
          this as a lean, not a lock. Ratings are chronological Elo.
        </p>
      </section>

      <section className="output card center">
        {res ? (
          <>
            <div className="versus">
              <div className="vs-team">
                <Gauge pct={res.team_a_win_prob} team={a} label="win" />
                <div className="vs-name"><TeamBadge team={a} size={26} /> {a}</div>
                <div className="vs-elo">Elo {res.team_a_elo}</div>
              </div>
              <div className="vs-sep">vs</div>
              <div className="vs-team">
                <Gauge pct={res.team_b_win_prob} team={b} label="win" />
                <div className="vs-name"><TeamBadge team={b} size={26} /> {b}</div>
                <div className="vs-elo">Elo {res.team_b_elo}</div>
              </div>
            </div>
            <ProbBar leftTeam={a} rightTeam={b} leftPct={res.team_a_win_prob} />
          </>
        ) : <div className="loading">Pick two teams…</div>}
      </section>
    </div>
  );
}
