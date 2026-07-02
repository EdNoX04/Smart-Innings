// API base: in dev, Vite proxies /api -> http://localhost:8000.
// Override with VITE_API_URL when deploying frontend and backend separately.
const BASE = import.meta.env.VITE_API_URL || "";

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const getMeta = async () => {
  const res = await fetch(`${BASE}/api/meta`);
  if (!res.ok) throw new Error("Could not reach the SmartInnings API");
  return res.json();
};

export const predictChase = (b) => post("/api/predict/chase", b);
export const predictChaseSweep = (b) => post("/api/predict/chase/sweep", b);
export const predictScore = (b) => post("/api/predict/score", b);
export const predictPrematch = (b) => post("/api/predict/prematch", b);

export const startRefresh = async () => {
  const res = await fetch(`${BASE}/api/refresh`, { method: "POST" });
  if (!res.ok) throw new Error("Could not start the update");
  return res.json();
};

export const refreshStatus = async () => {
  const res = await fetch(`${BASE}/api/refresh/status`);
  if (!res.ok) throw new Error("Could not read update status");
  return res.json();
};
