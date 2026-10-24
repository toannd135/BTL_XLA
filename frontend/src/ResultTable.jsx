import { Progress, Table } from 'antd'
import { FruitSticker, StateMark } from './Sticker'
import { pct } from './labels'

const columns = [
  { title: '#', dataIndex: 'idx', width: 44 },
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

export default function ResultTable({ objects }) {
  return (
    <Table
      size="small"
      rowKey="idx"
      pagination={false}
      columns={columns}
      dataSource={objects.map((o, i) => ({ ...o, idx: i + 1 }))}
      locale={{ emptyText: 'Không thấy quả nào. Thử hạ ngưỡng độ tin cậy hoặc dùng ảnh rõ hơn.' }}
    />
  )
}
