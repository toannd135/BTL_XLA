import { boxColor } from './boxStyle.js'

const FONT = "'Be Vietnam Pro', system-ui, sans-serif"

// Vẽ đúng contour từ API. Spline có thể phình khỏi mask ở cuống và chỗ lõm.
function shapePath(ctx, o) {
  if (o.polygon && o.polygon.length >= 3) {
    ctx.beginPath()
    o.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
    ctx.closePath()
  } else if (o.bbox) {
    const [x1, y1, x2, y2] = o.bbox
    ctx.beginPath()
    ctx.rect(x1, y1, x2 - x1, y2 - y1)
  }
}

// Vẽ khung / mask + tem nhãn lên ctx, toạ độ theo pixel ảnh gốc
export function drawBoxes(ctx, objects, w, h, { selectedIndex = null, displayScale } = {}) {
  const s = displayScale ?? Math.max(0.35, Math.max(w, h) / 960)
  const selected = selectedIndex !== null && objects[selectedIndex] ? selectedIndex : null
  const usedLabels = []
  ctx.save()
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.textBaseline = 'middle'
  const order = objects.map((_, i) => i)
  if (selected !== null) order.push(...order.splice(order.indexOf(selected), 1))
  order.forEach((index) => {
    const o = objects[index]
    // Gắn tem nhãn vào đỉnh cao nhất của polygon hoặc góc trên-trái bbox
    const [x1, y1] = (o.polygon && o.polygon.length >= 3)
      ? o.polygon.reduce((top, p) => (p[1] < top[1] ? p : top))
      : (o.bbox || [0, 0])
    const active = selected === index
    const muted = selected !== null && !active
    const color = boxColor(index)

    // Tô vùng mask bán trong suốt cho Instance Segmentation để phân biệt rõ từng quả
    if (o.polygon && o.polygon.length >= 3) {
      ctx.save()
      ctx.fillStyle = color
      ctx.globalAlpha = active ? 0.35 : (muted ? 0.05 : 0.22)
      shapePath(ctx, o)
      ctx.fill()
      ctx.restore()
    }

    ctx.globalAlpha = muted ? 0.15 : 1
    // Viền đậm phía ngoài giữ độ tương phản trên mọi nền
    ctx.strokeStyle = 'rgba(0,0,0,0.55)'
    ctx.lineWidth = (active ? 5 : 3) * s
    if (o.has_mask_overlap ?? o.is_occluded) {
      ctx.setLineDash([4 * s, 4 * s])
    } else {
      ctx.setLineDash([])
    }
    shapePath(ctx, o)
    ctx.stroke()

    // Viền màu đại diện cho đối tượng
    ctx.strokeStyle = color
    ctx.lineWidth = (active ? 3 : 1.5) * s
    shapePath(ctx, o)
    ctx.stroke()
    ctx.setLineDash([])

    if (active && (!o.polygon || o.polygon.length < 3)) {
      ctx.globalAlpha = 0.08
      ctx.fillStyle = color
      ctx.fill()
      ctx.globalAlpha = 1
    }
    if (muted) return

    // Tem nhãn định danh đồng bộ với bảng kết quả
    const label = `#${index + 1}`
    ctx.font = `700 ${12 * s}px ${FONT}`
    const pad = 5 * s
    const bw = ctx.measureText(label).width + pad * 2
    const bh = 21 * s
    let tx = Math.max(0, Math.min(x1 + 3 * s, w - bw))
    let ty = Math.max(0, Math.min(y1 + 3 * s, h - bh))
    for (let attempt = 0; attempt < 20; attempt++) {
      const x = Math.max(0, Math.min(x1 + 3 * s + (attempt % 4) * (bw + 3 * s), w - bw))
      const y = Math.max(0, Math.min(y1 + 3 * s + Math.floor(attempt / 4) * (bh + 3 * s), h - bh))
      tx = x
      ty = y
      if (!usedLabels.some((r) => x < r.x + r.w + s && x + bw + s > r.x && y < r.y + r.h + s && y + bh + s > r.y)) break
    }
    usedLabels.push({ x: tx, y: ty, w: bw, h: bh })
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.roundRect(tx, ty, bw, bh, 3 * s)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.fillText(label, tx + pad, ty + bh / 2)
  })
  ctx.restore()
}

export function downloadCanvas(canvas, name) {
  canvas.toBlob((blob) => {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = name
    a.click()
    URL.revokeObjectURL(a.href)
  }, 'image/jpeg', 0.92)
}
