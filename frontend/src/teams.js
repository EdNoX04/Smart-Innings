// Team display metadata: short code + brand colors used across the UI.
export const TEAM_META = {
  "Chennai Super Kings":         { code: "CSK",  primary: "#F9CD05", text: "#1f2a44" },
  "Mumbai Indians":              { code: "MI",   primary: "#045093", text: "#ffffff" },
  "Royal Challengers Bengaluru": { code: "RCB",  primary: "#D5152D", text: "#ffffff" },
  "Kolkata Knight Riders":       { code: "KKR",  primary: "#3A225D", text: "#F2C94C" },
  "Sunrisers Hyderabad":         { code: "SRH",  primary: "#F26522", text: "#1f2a44" },
  "Delhi Capitals":              { code: "DC",   primary: "#17449B", text: "#ffffff" },
  "Punjab Kings":                { code: "PBKS", primary: "#D71920", text: "#ffffff" },
  "Rajasthan Royals":            { code: "RR",   primary: "#EA1A85", text: "#ffffff" },
  "Gujarat Titans":              { code: "GT",   primary: "#1B2133", text: "#C9A24B" },
  "Lucknow Super Giants":        { code: "LSG",  primary: "#0D4DA1", text: "#67E8F9" },
};

export const meta = (team) =>
  TEAM_META[team] || { code: (team || "?").slice(0, 3).toUpperCase(), primary: "#475569", text: "#fff" };
