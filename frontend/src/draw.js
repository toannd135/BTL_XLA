import { boxColor } from './boxStyle'

const FONT = "'Be Vietnam Pro', system-ui, sans-serif"

// khung xoay (model OBB) nếu có polygon, ngược lại box thẳng
function shapePath(ctx, o) {
  ctx.beginPath()
  if (o.polygon?.length === 4) {
    o.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
    ctx.closePath()
  } else {
    const [x1, y1, x2, y2] = o.bbox
    ctx.rect(x1, y1, x2 - x1, y2 - y1)
  }
}

// vẽ khung + tem nhãn lên ctx, toạ độ theo pixel ảnh gốc
export function drawBoxes(ctx, objects, w, h, { selectedIndex = null, displayScale } = {}) {
  const s = displayScale ?? Math.max(0.35, Math.max(w, h) / 960)
  const selected = selectedIndex !== null && objects[selectedIndex] ? selectedIndex : null
  const usedLabels = []
  ctx.save()
  ctx.textBaseline = 'middle'
  const order = objects.map((_, i) => i)
  if (selected !== null) order.push(...order.splice(order.indexOf(selected), 1))
  order.forEach((index) => {
    const o = objects[index]
    // Khung xoay: góc trên-trái của bbox bao ngoài nằm xa khung (khoảng trống), nên gắn tem vào đỉnh cao nhất của polygon.
    const [x1, y1] = o.polygon?.length === 4
      ? o.polygon.reduce((top, p) => (p[1] < top[1] ? p : top))
      : o.bbox
    const active = selected === index
    const muted = selected !== null && !active
    const color = boxColor(index)
    ctx.globalAlpha = muted ? 0.15 : 1
    // A thin dark outline keeps bright borders visible on both white and yellow fruit.
    ctx.strokeStyle = 'rgba(0,0,0,0.55)'
    ctx.lineWidth = (active ? 5 : 3) * s
    shapePath(ctx, o)
    ctx.stroke()
    ctx.strokeStyle = color
    ctx.lineWidth = (active ? 3 : 1.5) * s
    ctx.stroke()
    if (active) {
      ctx.globalAlpha = 0.08
      ctx.fillStyle = color
      ctx.fill()
      ctx.globalAlpha = 1
    }
    if (muted) return

    // Compact IDs match the table; stagger badges rather than covering fruit with full labels.
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
