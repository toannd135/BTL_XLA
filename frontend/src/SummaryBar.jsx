import { STATE_COLOR, UNSURE_COLOR } from './labels'

const PARTS = [
  { key: 'fresh', text: 'Tươi', color: STATE_COLOR.fresh },
  { key: 'rotten', text: 'Hỏng', color: STATE_COLOR.rotten },
  { key: 'unsure', text: 'Chưa chắc', color: UNSURE_COLOR },
]

// thanh tỉ lệ tươi / hỏng / chưa chắc kèm số đếm
export default function SummaryBar({ c, note }) {
  return (
    <div className="summary">
      <div className="summary-head">
        <span className="summary-total">{c.total} quả</span>
        {note && <span className="summary-note">{note}</span>}
      </div>
      <div className="summary-bar">
        {c.total === 0 ? (
          <span style={{ flex: 1, background: '#e1e8dc' }} />
        ) : (
          PARTS.filter((p) => c[p.key] > 0).map((p) => (
            <span key={p.key} style={{ flex: c[p.key], background: p.color }} />
          ))
        )}
      </div>
      <div className="summary-legend">
        {PARTS.map((p) => (
          <span key={p.key}>
            <i style={{ background: p.color }} />
            {p.text} <b>{c[p.key]}</b>
          </span>
        ))}
      </div>
    </div>
  )
}
