import { forwardRef, useEffect, useRef, useImperativeHandle } from 'react'
import { drawBoxes } from './draw'

// Vẽ ảnh gốc lên canvas đúng kích thước thật, CSS co lại cho vừa khung
const ImageBoxes = forwardRef(function ImageBoxes({ src, result, objects }, outer) {
  const ref = useRef(null)
  useImperativeHandle(outer, () => ref.current)

  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      const canvas = ref.current
      if (!canvas) return
      const w = result?.image_width || img.naturalWidth
      const h = result?.image_height || img.naturalHeight
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, w, h)
      // đợi font web tải xong để chữ trên tem không bị font dự phòng
      document.fonts.ready.then(() => drawBoxes(ctx, objects, w, h))
    }
    img.src = src
  }, [src, result, objects])

  return <canvas ref={ref} className="preview" />
})

export default ImageBoxes
