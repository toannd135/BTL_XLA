export const ACCEPT = ['image/jpeg', 'image/png', 'image/webp']
export const MAX_MB = 10
export const MAX_FILES = 50
// nhận cả video ở ô thả ảnh, vì antd lặng lẽ bỏ qua file không khớp accept khi kéo thả
export const DROP_ACCEPT = '.jpg,.jpeg,.png,.webp,video/*'
export const isVideo = (f) => f.type.startsWith('video/')

// kiểm tra file trước khi gửi, trả về thông báo lỗi hoặc null
export function checkFile(f) {
  if (!ACCEPT.includes(f.type)) return 'Chỉ nhận ảnh jpg, png hoặc webp'
  if (f.size > MAX_MB * 1024 * 1024) return `Ảnh vượt quá ${MAX_MB}MB`
  return null
}
