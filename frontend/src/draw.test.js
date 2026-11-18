import assert from 'node:assert/strict'
import test from 'node:test'
import { drawBoxes } from './draw.js'

function context() {
  const paths = []
  return {
    paths, save() {}, restore() {}, fill() {}, stroke() {}, closePath() {},
    setLineDash() {}, roundRect() {}, fillText() {},
    measureText: () => ({ width: 12 }),
    beginPath() { paths.push([]) },
    moveTo(x, y) { paths.at(-1).push([x, y]) },
    lineTo(x, y) { paths.at(-1).push([x, y]) },
    rect(...values) { paths.at(-1).push(values) },
    bezierCurveTo() { assert.fail('A mask contour must not be moved by a spline') },
  }
}

test('concave masks follow the exact backend contour for fill and both strokes', () => {
  const ctx = context()
  const polygon = [[10, 10], [90, 10], [90, 90], [50, 50], [10, 90]]
  drawBoxes(ctx, [{ polygon, bbox: [10, 10, 90, 90] }], 100, 100)
  assert.deepEqual(ctx.paths.slice(0, 3), [polygon, polygon, polygon])
  assert.equal(ctx.lineJoin, 'round')
})

test('missing masks retain a rectangular detection fallback', () => {
  const ctx = context()
  drawBoxes(ctx, [{ bbox: [10, 20, 40, 70] }], 100, 100)
  assert.deepEqual(ctx.paths[0], [[10, 20, 30, 50]])
})
