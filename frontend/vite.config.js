import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/analytics': 'http://127.0.0.1:8000',
      '/uploads': 'http://127.0.0.1:8000',
      '/sessions': 'http://127.0.0.1:8000',
      '/api/admin': 'http://127.0.0.1:8000',
      '/api/orders': 'http://127.0.0.1:8000',
    }
  },
  build: {
    outDir: 'dist',
  }
})
