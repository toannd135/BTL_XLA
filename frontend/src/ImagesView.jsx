import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert, App as AntApp, Button, Card, Col, Empty, Progress, Row, Space, Spin, Tag, Typography, Upload,
} from 'antd'
import { predict, errorText } from './api'
import ImageBoxes from './ImageBoxes'
import ResultTable from './ResultTable'
import SummaryBar from './SummaryBar'
import { downloadCanvas } from './draw'
import { checkFile, isVideo, DROP_ACCEPT, MAX_FILES, MAX_MB } from './files'
import { count, passConf, STATE_COLOR } from './labels'
import { boxColor } from './boxStyle'

const STATUS = {
  wait: <Tag>Chờ</Tag>,
  run: <Tag color="blue">Đang chạy</Tag>,
  error: <Tag color="red">Lỗi</Tag>,
}

// phần tử tự dùng phím mũi tên
const ARROW_OWNERS = 'input, textarea, select, video, [contenteditable="true"], [role="slider"], ' +
  '[role="tab"], [role="radio"], [role="switch"], .ant-slider, .ant-segmented'

// tên file bắt đầu bằng = + - @ bị Excel hiểu là công thức, thêm ' phía trước để giữ nguyên chữ
const csvText = (s) => `"${(/^[=+\-@\t\r]/.test(s) ? "'" + s : s).replaceAll('"', '""')}"`

const filterObjects = (result, minConf, excluded = []) =>
  (result?.objects || []).filter((o, i) => passConf(o, minConf) && !excluded.includes(i))

// ảnh thu nhỏ + khung theo % kích thước ảnh, nhẹ hơn vẽ canvas cỡ gốc cho từng ảnh
function Thumb({ row, active, onOpen, onRemove }) {
  const res = row.result
  return (
    <div className={'g-item' + (active ? ' active' : '')} onClick={onOpen} title={row.error || row.name}>
      <div className="g-img">
        <img src={row.url} alt={row.name} loading="lazy" />
        {res && row.objects.map((o, i) => {
          const [x1, y1, x2, y2] = o.bbox
          const color = boxColor(i)
          return (
            <span key={i} className="g-box" style={{
              left: `${(x1 / res.image_width) * 100}%`,
              top: `${(y1 / res.image_height) * 100}%`,
              width: `${((x2 - x1) / res.image_width) * 100}%`,
              height: `${((y2 - y1) / res.image_height) * 100}%`,
              borderColor: color,
            }}>
              <span className="g-tag" style={{ background: color }}>
                #{i + 1}
              </span>
            </span>
          )
        })}
        <Button className="g-remove" type="text" size="small" title="Bỏ ảnh này"
          onClick={(e) => { e.stopPropagation(); onRemove() }}>
          ×
        </Button>
      </div>
      <div className="g-meta">
        <span className="g-name">{row.name}</span>
        {row.status === 'done' ? (
          <span className="g-count">
            <b style={{ color: STATE_COLOR.fresh }}>{row.c.fresh}</b> tươi,{' '}
            <b style={{ color: STATE_COLOR.rotten }}>{row.c.rotten}</b> hỏng
          </span>
        ) : STATUS[row.status]}
      </div>
    </div>
  )
}

export default function ImagesView({ minConf, onVideo }) {
  const { message } = AntApp.useApp()
  const [items, setItems] = useState([])
  const [selectedKey, setSelectedKey] = useState(null)
  const [selectedBox, setSelectedBox] = useState(null)
  const [running, setRunning] = useState(false)
  const [stopping, setStopping] = useState(false)
  const canvas = useRef(null)
  const box = useRef(null)
  const stopRef = useRef(false)
  const runningRef = useRef(false)
  // itemsRef luôn là bản mới nhất để vòng lặp nhận dạng đọc được ngay, không chờ render
  const itemsRef = useRef(items)

  const commit = (next) => {
    itemsRef.current = next
    setItems(next)
  }
  const update = (key, patch) =>
    commit(itemsRef.current.map((it) => (it.key === key ? { ...it, ...patch } : it)))

  // rời trang: dừng hàng đợi và thu hồi object url
  useEffect(() => () => {
    stopRef.current = true
    itemsRef.current.forEach((it) => URL.revokeObjectURL(it.url))
  }, [])

  // hàng đợi tuần tự: mỗi lần một ảnh, lấy ảnh 'wait' kế tiếp cho tới khi hết hoặc bị dừng
  const pump = async () => {
    stopRef.current = false
    setStopping(false)
    if (runningRef.current) return
    runningRef.current = true
    setRunning(true)
    while (!stopRef.current) {
      const it = itemsRef.current.find((x) => x.status === 'wait')
      if (!it) break
      update(it.key, { status: 'run', error: null })
      try {
        const data = await predict(it.file)
        // ảnh đã bị bỏ trong lúc chờ thì update không tìm thấy key, kết quả tự bị bỏ qua
        update(it.key, { status: 'done', result: data, error: null, excluded: [] })
      } catch (err) {
        update(it.key, { status: 'error', error: errorText(err) })
      }
    }
    runningRef.current = false
    setRunning(false)
    setStopping(false)
  }

  const add = (f, list) => {
    // antd gọi beforeUpload cho từng file, chỉ xử lý cả lượt 1 lần ở file cuối
    if (f !== list[list.length - 1]) return Upload.LIST_IGNORE

    const videos = list.filter(isVideo)
    const images = list.filter((x) => !isVideo(x))
    images.forEach((x) => {
      const err = checkFile(x)
      if (err) message.error(`${x.name}: ${err}`)
    })
    const ok = images.filter((x) => !checkFile(x))
    if (videos.length && !ok.length) {
      onVideo(videos[0])
      return Upload.LIST_IGNORE
    }
    if (videos.length) message.warning(`Bỏ qua ${videos.length} video, tab này chỉ nhận ảnh`)

    const old = itemsRef.current
    const room = MAX_FILES - old.length
    if (ok.length > room) message.warning(`Chỉ nhận tối đa ${MAX_FILES} ảnh`)
    const added = ok.slice(0, Math.max(0, room)).map((x) => ({
      key: x.uid, name: x.name, file: x, url: URL.createObjectURL(x), status: 'wait',
      result: null, error: null,
    }))
    if (!added.length) return Upload.LIST_IGNORE
    commit([...old, ...added])
    if (!old.some((it) => it.key === selectedKey)) setSelectedKey(added[0].key)
    pump()
    return Upload.LIST_IGNORE
  }

  const remove = (key) => {
    const list = itemsRef.current
    const i = list.findIndex((x) => x.key === key)
    if (i < 0) return
    URL.revokeObjectURL(list[i].url)
    if (key === selectedKey) setSelectedKey((list[i + 1] || list[i - 1])?.key ?? null)
    commit(list.filter((x) => x.key !== key))
  }

  const clear = () => {
    itemsRef.current.forEach((it) => URL.revokeObjectURL(it.url))
    commit([])
    setSelectedKey(null)
  }

  const stop = () => {
    stopRef.current = true
    setStopping(true)
  }

  const retry = (key) => {
    commit(itemsRef.current.map((it) =>
      (it.status === 'error' && (!key || it.key === key) ? { ...it, status: 'wait', error: null } : it)))
    pump()
  }

  const rerunSelected = () => {
    if (!selectedKey || runningRef.current) return
    update(selectedKey, { status: 'wait', error: null, result: null, excluded: [] })
    pump()
  }

  const recognitionInfo = (
    <div className="image-mode">
      <Space wrap>
        <span>Nhận dạng ảnh</span>
        <Tag color="blue">Kiểm tra kỹ</Tag>
      </Space>
      <Typography.Text type="secondary">
        Hãy đối chiếu các khung với từng quả, đặc biệt khi quả nằm sát nhau hoặc che khuất.
      </Typography.Text>
    </div>
  )

  const rows = useMemo(() => items.map((it) => {
    const objects = filterObjects(it.result, minConf, it.excluded)
    return { ...it, objects, c: count(objects) }
  }), [items, minConf])
  const done = rows.filter((r) => r.status === 'done')
  const waiting = rows.some((r) => r.status === 'wait')
  const failed = rows.some((r) => r.status === 'error')
  const sum = (k) => done.reduce((s, r) => s + r.c[k], 0)

  const index = rows.findIndex((r) => r.key === selectedKey)
  const sel = rows[index]
  // giữ mảng khung ổn định giữa các lần render để canvas không vẽ lại thừa
  const selResult = sel?.status === 'done' ? sel.result : null
  const selExcluded = sel?.excluded
  const selObjects = useMemo(() => filterObjects(selResult, minConf, selExcluded), [selResult, minConf, selExcluded])
  useEffect(() => { setSelectedBox(null) }, [selectedKey, selResult, minConf, selExcluded])
  const selectBox = (i) => setSelectedBox((current) => current === i ? null : i)
  const excludeSelectedBox = () => {
    if (selectedBox === null || !selResult) return
    const object = selObjects[selectedBox]
    const originalIndex = selResult.objects.indexOf(object)
    if (originalIndex < 0 || sel.excluded?.includes(originalIndex)) return
    update(sel.key, { excluded: [...(sel.excluded || []), originalIndex] })
    setSelectedBox(null)
  }

  const go = (step) => {
    const next = rows[index + step]
    if (next) setSelectedKey(next.key)
  }

  // phím mũi tên trái/phải để chuyển ảnh, bỏ qua khi tab đang ẩn hoặc khi phím đang điều khiển
  // một ô khác (ô nhập, thanh ngưỡng, nút chọn, thanh tab) để không vừa đổi ngưỡng vừa chuyển ảnh
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return
      if (document.activeElement?.closest?.(ARROW_OWNERS)) return
      if (!box.current || box.current.offsetParent === null) return
      go(e.key === 'ArrowLeft' ? -1 : 1)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const exportCsv = () => {
    const head = 'ten_file,so_qua,tuoi,hong,chua_chac,thoi_gian_ms,nguong_tin_cay,che_do,khung_loai_thu_cong'
    const lines = done.map((r) =>
      [csvText(r.name), r.c.total, r.c.fresh, r.c.rotten, r.c.unsure,
        r.result.inference_ms, minConf, r.result.mode || 'careful', r.excluded?.length || 0].join(','))
    // thêm BOM để Excel đọc đúng tiếng Việt
    const blob = new Blob(['﻿' + [head, ...lines].join('\n')], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'ket_qua_nhan_dang.csv'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const download = () => {
    const name = sel.name.replace(/\.[^.]+$/, '') + '_ketqua.jpg'
    downloadCanvas(canvas.current, name)
  }

  if (items.length === 0) {
    return (
      <div ref={box}>
        <Card size="small">
          {recognitionInfo}
          <Upload.Dragger className="drop-big" multiple accept={DROP_ACCEPT}
            showUploadList={false} beforeUpload={add}>
            <p className="ant-upload-text">Thả một hoặc nhiều ảnh vào đây, hoặc bấm để chọn</p>
            <p className="ant-upload-hint">
              JPG, PNG hoặc WebP, tối đa {MAX_FILES} ảnh, mỗi ảnh không quá {MAX_MB}MB
            </p>
          </Upload.Dragger>
        </Card>
      </div>
    )
  }

  const selC = count(selObjects)

  let resultBody
  if (!sel) {
    resultBody = <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Chọn một ảnh để xem kết quả" />
  } else if (sel.status === 'wait') {
    resultBody = <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Đang chờ nhận dạng" />
  } else if (sel.status === 'run') {
    resultBody = (
      <div className="img-busy">
        <Spin />
        <span>Đang nhận dạng…</span>
      </div>
    )
  } else if (sel.status === 'error') {
    resultBody = (
      <Alert type="error" showIcon message={sel.error}
        action={<Button size="small" onClick={() => retry(sel.key)} disabled={running}>Thử lại</Button>} />
    )
  } else {
    resultBody = (
      <>
        <SummaryBar c={selC}
          note={`${sel.result.inference_ms} ms, ảnh ${sel.result.image_width}×${sel.result.image_height}`} />
        <Typography.Paragraph type="secondary" className="box-hint">
          Số và màu khớp với khung trên ảnh. Bấm một dòng để làm nổi khung hoặc loại khung sai.
        </Typography.Paragraph>
        {selectedBox !== null && (
          <Space wrap className="mb16">
            <Button size="small" onClick={() => setSelectedBox(null)}>Hiện tất cả khung</Button>
            <Button size="small" danger onClick={excludeSelectedBox}>Loại khung #{selectedBox + 1} khỏi kết quả</Button>
          </Space>
        )}
        {!!sel.excluded?.length && (
          <Typography.Paragraph type="secondary">
            Đã loại thủ công {sel.excluded.length} khung.{' '}
            <Button type="link" size="small" onClick={() => update(sel.key, { excluded: [] })}>
              Khôi phục các khung đã loại
            </Button>
          </Typography.Paragraph>
        )}
        <ResultTable objects={selObjects} selectedIndex={selectedBox} onSelect={selectBox} />
      </>
    )
  }

  return (
    <div ref={box}>
      <Card size="small" className="mb16">
        {recognitionInfo}
        <div className="img-toolbar">
          <Upload.Dragger className="drop-small" multiple accept={DROP_ACCEPT}
            showUploadList={false} beforeUpload={add}>
            <p className="ant-upload-text">Thêm ảnh</p>
            <p className="ant-upload-hint">Thả thêm ảnh hoặc bấm để chọn</p>
          </Upload.Dragger>

          <div className="img-actions">
            <Space wrap>
              <Button onClick={rerunSelected} disabled={running || !sel}>Nhận dạng lại ảnh này</Button>
              {running && (
                <Button danger onClick={stop} loading={stopping}>
                  {stopping ? 'Đang dừng…' : 'Dừng'}
                </Button>
              )}
              {!running && waiting && (
                <Button type="primary" onClick={pump}>Tiếp tục nhận dạng</Button>
              )}
              {failed && (
                <Button onClick={() => retry()} disabled={running}>Thử lại ảnh lỗi</Button>
              )}
              <Button onClick={exportCsv} disabled={running || done.length === 0}>Xuất CSV</Button>
              <Button danger onClick={clear} disabled={running}>Xoá tất cả</Button>
            </Space>
            <div className="img-progress">
              <span>Đã nhận dạng {done.length}/{items.length}</span>
              <Progress size="small" showInfo={false}
                percent={Math.round((done.length / items.length) * 100)} />
            </div>
          </div>
        </div>

        {items.length > 1 && done.length > 0 && (
          <div className="mt16">
            <SummaryBar note={`Từ ${done.length} ảnh`}
              c={{ total: sum('total'), fresh: sum('fresh'), rotten: sum('rotten'), unsure: sum('unsure') }} />
          </div>
        )}
      </Card>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <Card size="small"
            title={sel && (
              <div className="viewer-title">
                <span className="viewer-name">{sel.name}</span>
                <Typography.Text type="secondary">{index + 1}/{rows.length}</Typography.Text>
              </div>
            )}
            extra={(
              <Space size={4}>
                <Button size="small" onClick={() => go(-1)} disabled={index <= 0} title="Ảnh trước">‹</Button>
                <Button size="small" onClick={() => go(1)} disabled={index < 0 || index >= rows.length - 1}
                  title="Ảnh sau">›</Button>
                <Button size="small" onClick={download} disabled={sel?.status !== 'done'}>Tải ảnh kết quả</Button>
              </Space>
            )}
          >
            {sel ? (
              <Spin spinning={sel.status === 'run'}>
                <ImageBoxes ref={canvas} src={sel.url} result={selResult} objects={selObjects} selectedIndex={selectedBox} />
              </Spin>
            ) : (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Chọn một ảnh ở danh sách bên dưới" />
            )}
          </Card>
        </Col>

        <Col xs={24} xl={8}>
          <Card title="Kết quả" size="small">{resultBody}</Card>
        </Col>
      </Row>

      {items.length > 1 && (
        <Card title={`Ảnh đã chọn (${items.length})`} size="small" className="mt16">
          <div className="gallery">
            {rows.map((r) => (
              <Thumb key={r.key} row={r} active={r.key === selectedKey}
                onOpen={() => setSelectedKey(r.key)} onRemove={() => remove(r.key)} />
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
