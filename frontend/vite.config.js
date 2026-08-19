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
  build: {
    // The remaining warning past this point is the recharts vendor chunk
    // itself (~526 kB unminified-gzip-adjacent) — it no longer ships with the
    // initial page load (that dropped to ~155 kB across index + vendor), it
    // only loads once a run exists and a chart actually renders. Raising the
    // limit here says so explicitly instead of leaving a warning that reads
    // as unfixed.
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        // recharts (and its d3 dependencies) is the one dependency big enough
        // to matter — pulling it into its own vendor chunk means it caches
        // independently of app code across deploys, on top of already being
        // lazy-loaded behind Dashboard/Workspace.
        manualChunks: {
          recharts: ["recharts"],
        },
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
