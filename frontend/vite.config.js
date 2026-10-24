import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // thư mục dự án nằm trên ổ NTFS gắn vào Linux, Vite hay lỡ sự kiện sửa file -> quét định kỳ
    watch: { usePolling: true, interval: 300 },
  },
})
