import { FRUIT, FRUIT_COLOR, stateColor, stateText } from './labels'

const FONT = "'Be Vietnam Pro', system-ui, sans-serif"

// vẽ khung + tem nhãn lên ctx, toạ độ theo pixel ảnh gốc
export function drawBoxes(ctx, objects, w, h) {
  const s = Math.max(1, Math.max(w, h) / 640)
  ctx.textBaseline = 'middle'
  objects.forEach((o) => {
    const [x1, y1, x2, y2] = o.bbox
    const color = stateColor(o)
    ctx.strokeStyle = color
    ctx.lineWidth = 3 * s
    ctx.beginPath()
    ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 6 * s)
    ctx.stroke()

    // tem: phần tên quả màu theo loại quả, phần tình trạng màu theo tươi/hỏng
    const fruit = FRUIT[o.label] || o.label
    const state = stateText(o)
    const fc = FRUIT_COLOR[o.label] || { bg: '#5b6b5c', fg: '#fff' }
    const th = 22 * s
    const pad = 9 * s
    ctx.font = `800 ${12 * s}px ${FONT}`
    const fw = ctx.measureText(fruit).width + pad * 2
    ctx.font = `600 ${12 * s}px ${FONT}`
    const sw = ctx.measureText(state).width + pad * 2
    const tx = Math.min(x1, w - fw - sw)
    const ty = y1 - th - 4 * s >= 0 ? y1 - th - 4 * s : y1 + 4 * s

    ctx.fillStyle = color
    ctx.beginPath()
    ctx.roundRect(tx, ty, fw + sw, th, th / 2)
    ctx.fill()
    ctx.fillStyle = fc.bg
    ctx.beginPath()
    ctx.roundRect(tx, ty, fw, th, th / 2)
    ctx.fill()

    ctx.fillStyle = fc.fg
    ctx.font = `800 ${12 * s}px ${FONT}`
    ctx.fillText(fruit, tx + pad, ty + th / 2)
    ctx.fillStyle = '#fff'
    ctx.font = `600 ${12 * s}px ${FONT}`
    ctx.fillText(state, tx + fw + pad, ty + th / 2)
  })
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
