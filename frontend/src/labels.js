export const FRUIT = { banana: 'Chuối', apple: 'Táo', orange: 'Cam' }
export const STATE = { fresh: 'Tươi', rotten: 'Hỏng' }

// màu nhận diện từng loại quả; chữ tối trên nền vàng cho dễ đọc
export const FRUIT_COLOR = {
  banana: { bg: '#f2c230', fg: '#3d2f00' },
  apple: { bg: '#d83b2d', fg: '#ffffff' },
  orange: { bg: '#f08a24', fg: '#ffffff' },
}
export const STATE_COLOR = { fresh: '#2e8b47', rotten: '#7a4a2e' }

// dưới mức này coi như classifier chưa chắc tươi hay hỏng
export const UNSURE = 0.6
export const UNSURE_COLOR = '#8a8f87'

export const pct = (v) => Math.round(v * 100)
export const isUnsure = (o) => o.state_confidence < UNSURE

export const stateOf = (o) => (isUnsure(o) ? 'unsure' : o.state)
export const stateColor = (o) => (isUnsure(o) ? UNSURE_COLOR : STATE_COLOR[o.state] || UNSURE_COLOR)
export const stateText = (o) => (isUnsure(o) ? 'Chưa chắc' : STATE[o.state] || o.state)

export function count(objects) {
  const fresh = objects.filter((o) => stateOf(o) === 'fresh').length
  const rotten = objects.filter((o) => stateOf(o) === 'rotten').length
  return { total: objects.length, fresh, rotten, unsure: objects.length - fresh - rotten }
}

// Box "rescued": điểm detector dưới ngưỡng server nhưng CLIP đã xác nhận là quả -> tính như vừa đạt ngưỡng mặc định 0.4
export const passConf = (o, minConf) => (o.rescued ? Math.max(o.confidence, 0.4) : o.confidence) >= minConf
