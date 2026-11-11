import { Progress, Table } from 'antd'
import { FruitSticker, StateMark } from './Sticker'
import { pct } from './labels'
import { boxColor } from './boxStyle'

const columns = [
  { title: '#', dataIndex: 'idx', width: 52,
    render: (v) => <span className="box-id" style={{ background: boxColor(v - 1) }}>#{v}</span> },
  {
    title: 'Loại quả',
    dataIndex: 'label',
    render: (v) => <FruitSticker label={v} />,
  },
  {
    title: 'Độ tin cậy',
    dataIndex: 'confidence',
    width: 140,
    render: (v) => <Progress percent={pct(v)} size="small" strokeColor="#2e6b3a" />,
  },
  {
    title: 'Tình trạng',
    dataIndex: 'state',
    render: (_, r) => <StateMark obj={r} />,
  },
]

export default function ResultTable({ objects, selectedIndex = null, onSelect }) {
  return (
    <Table
      size="small"
      rowKey="idx"
      pagination={false}
      columns={columns}
      rowClassName={(r) => onSelect ? `box-row${r.idx - 1 === selectedIndex ? ' box-row-selected' : ''}` : ''}
      onRow={(r) => onSelect ? {
        onClick: () => onSelect(r.idx - 1),
        onKeyDown: (event) => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(r.idx - 1) }
        },
        tabIndex: 0,
        'aria-label': `Khung số ${r.idx}, bấm để làm nổi bật trên ảnh`,
        'aria-selected': r.idx - 1 === selectedIndex,
      } : {}}
      dataSource={objects.map((o, i) => ({ ...o, idx: i + 1 }))}
      locale={{ emptyText: 'Không thấy quả nào. Thử hạ ngưỡng độ tin cậy hoặc dùng ảnh rõ hơn.' }}
    />
  )
}
