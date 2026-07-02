import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // forward API calls to the FastAPI backend.
      // Use 127.0.0.1 (not "localhost"): Node resolves localhost to IPv6 ::1,
      // but uvicorn binds IPv4 127.0.0.1 -> ECONNREFUSED ::1:8000.
      "/api": "http://127.0.0.1:8000",
    },
  },
});
