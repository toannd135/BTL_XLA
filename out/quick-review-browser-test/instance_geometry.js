/* Normalized review-box edits, shared by the offline page and Node tests. */
;(function (root) {
  'use strict'
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v))
  function corners(box) {
    const [, x, y, w, h] = box
    return { nw: { x: x-w/2, y: y-h/2 }, ne: { x: x+w/2, y: y-h/2 },
      se: { x: x+w/2, y: y+h/2 }, sw: { x: x-w/2, y: y+h/2 } }
  }
  function move(box, origin, point) {
    const [cls, x, y, w, h] = box
    return [cls, clamp(x+point.x-origin.x, w/2, 1-w/2), clamp(y+point.y-origin.y, h/2, 1-h/2), w, h]
  }
  function resize(box, corner, point, minimum = .002) {
    if (!['nw','ne','se','sw'].includes(corner)) throw Error('Invalid corner')
    const c = corners(box), west = corner.includes('w'), north = corner.includes('n')
    const fixed = c[{nw:'se',ne:'sw',se:'nw',sw:'ne'}[corner]]
    const minX = Math.min(minimum, west ? fixed.x : 1-fixed.x)
    const minY = Math.min(minimum, north ? fixed.y : 1-fixed.y)
    const px = west ? clamp(point.x, 0, fixed.x-minX) : clamp(point.x, fixed.x+minX, 1)
    const py = north ? clamp(point.y, 0, fixed.y-minY) : clamp(point.y, fixed.y+minY, 1)
    return [box[0], (px+fixed.x)/2, (py+fixed.y)/2, Math.abs(px-fixed.x), Math.abs(py-fixed.y)]
  }
  function handle(box, point, toleranceX, toleranceY) {
    return Object.entries(corners(box)).find(([, p]) =>
      Math.abs(p.x-point.x) <= toleranceX && Math.abs(p.y-point.y) <= toleranceY)?.[0] ?? null
  }
  const api = { corners, move, resize, handle }
  if (typeof module !== 'undefined' && module.exports) module.exports = api
  else root.InstanceGeometry = api
})(typeof window !== 'undefined' ? window : globalThis)
