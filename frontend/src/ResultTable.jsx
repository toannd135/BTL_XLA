import { Progress, Select, Table, Tag, Tooltip } from 'antd'
import { FruitSticker, StateMark } from './Sticker'
import { FRUIT, STATE, pct } from './labels'
import { boxColor } from './boxStyle'

const columns = [
  {
    title: '#',
    dataIndex: 'idx',
    width: 48,
    render: (v) => <span className="box-id" style={{ background: boxColor(v - 1) }}>#{v}</span>,
  },
  {
    title: 'Loại quả',
    dataIndex: 'label',
    render: (v, r) => r.onEdit ? (
      <Select
        size="small"
        value={v}
        options={Object.keys(FRUIT).map((value) => ({ value, label: FRUIT[value] }))}
        onChange={(value) => r.onEdit(r._source, 'label', value)}
        onClick={(event) => event.stopPropagation()}
      />
    ) : <FruitSticker label={v} />,
  },
  {
    title: 'Độ tin cậy',
    dataIndex: 'confidence',
    width: 130,
    render: (v) => <Progress percent={pct(v)} size="small" strokeColor="#2e6b3a" />,
  },
  {
    title: 'Tình trạng',
    dataIndex: 'state',
    render: (_, r) => r.onEdit ? (
      <Select
        size="small"
        value={r.state}
        options={Object.keys(STATE).map((value) => ({ value, label: STATE[value] }))}
        onChange={(value) => r.onEdit(r._source, 'state', value)}
        onClick={(event) => event.stopPropagation()}
      />
    ) : <StateMark obj={r} />,
  },
  {
    title: 'Chồng lấp mask',
    dataIndex: 'overlap_ratio',
    width: 120,
    render: (v, r) => {
      if (r.mask_valid === false) return <Tag color="warning">Thiếu mask</Tag>
      v = v ?? r.occlusion_ratio
      if (v === undefined || v === null) return <span style={{ color: '#aaa' }}>-</span>
      const pctVal = Math.round(v * 100)
      return (
        <Tooltip title="Tỉ lệ vùng mask trùng với mask quả khác; không đo phần quả bị che khuất.">
          <Tag color={pctVal >= 55 ? 'error' : pctVal >= 25 ? 'warning' : 'default'}>
            Trùng {pctVal}%
          </Tag>
        </Tooltip>
      )
    },
  },
]

export default function ResultTable({ objects, selectedIndex = null, onSelect, onEdit }) {
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
      dataSource={objects.map((o, i) => ({ ...o, _source: o, idx: i + 1, onEdit }))}
      locale={{ emptyText: 'Không thấy quả nào. Thử hạ ngưỡng độ tin cậy hoặc dùng ảnh rõ hơn.' }}
    />
  )
}
