import { FRUIT, FRUIT_COLOR, pct, stateColor, stateText } from './labels'

// tem dán hình bầu dục như tem trên trái cây ngoài chợ
export function FruitSticker({ label }) {
  const c = FRUIT_COLOR[label] || { bg: '#5b6b5c', fg: '#fff' }
  return (
    <span className="sticker" style={{ background: c.bg, color: c.fg }}>
      {FRUIT[label] || label}
    </span>
  )
}

export function StateMark({ obj }) {
  const color = stateColor(obj)
  return (
    <span className="state" style={{ color }}>
      <i style={{ background: color }} />
      {stateText(obj)}
      <small>{pct(obj.state_confidence)}%</small>
    </span>
  )
}
