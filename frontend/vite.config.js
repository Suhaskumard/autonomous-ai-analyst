import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the app and proxies /api to the local backend, so the
// browser never makes a cross-origin request and dev matches the container
// topology.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.js"],
    include: ["tests/**/*.test.jsx"],
  },
});
