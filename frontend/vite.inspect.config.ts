import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Temporary config for rendered inspection: proxies to the 8001 backend seeded with demo
// review artifacts, leaving the developer's own 3000/8000 pair untouched.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3002,
    strictPort: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8001', changeOrigin: true } },
  },
});
