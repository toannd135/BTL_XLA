import { forwardRef, useEffect, useRef, useImperativeHandle } from 'react'
import { drawBoxes } from './draw'

// Tọa độ theo ảnh gốc; tăng độ phân giải canvas khi ảnh nhỏ được phóng trên màn hình.
const ImageBoxes = forwardRef(function ImageBoxes({ src, result, objects, selectedIndex = null }, outer) {
  const ref = useRef(null)
  useImperativeHandle(outer, () => ref.current)

  useEffect(() => {
    let cancelled = false
    const img = new Image()
    const paint = async () => {
      await document.fonts.ready
      if (cancelled || !img.complete || !img.naturalWidth) return
      const canvas = ref.current
      if (!canvas) return
      const w = result?.image_width || img.naturalWidth
      const h = result?.image_height || img.naturalHeight
      const displayedWidth = canvas.getBoundingClientRect().width || w
      // Keep small source images' borders/text crisp when enlarged in the viewer.
      const backingScale = Math.max(1, Math.min(3, displayedWidth * (window.devicePixelRatio || 1) / w))
      canvas.width = Math.round(w * backingScale)
      canvas.height = Math.round(h * backingScale)
      const ctx = canvas.getContext('2d')
      ctx.setTransform(backingScale, 0, 0, backingScale, 0, 0)
      ctx.drawImage(img, 0, 0, w, h)
      drawBoxes(ctx, objects, w, h, { selectedIndex, displayScale: w / displayedWidth })
    }
    img.onload = paint
    img.src = src
    const observer = new ResizeObserver(paint)
    if (ref.current) observer.observe(ref.current)
    return () => { cancelled = true; observer.disconnect(); img.onload = null }
  }, [src, result, objects, selectedIndex])

  return <canvas ref={ref} className="preview" />
})

export default ImageBoxes
