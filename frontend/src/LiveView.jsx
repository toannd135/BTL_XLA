import { useEffect, useRef, useState } from 'react'
import {
  Alert, App as AntApp, Button, Card, Col, Empty, Row, Segmented, Select, Space, Switch, Upload,
} from 'antd'
import { predict, collectFrame, collectStats, errorText, isCanceled } from './api'
import ResultTable from './ResultTable'
import SummaryBar from './SummaryBar'
import { drawBoxes, downloadCanvas } from './draw'
import { count, FRUIT, STATE, passConf } from './labels'

const SPEEDS = [0.25, 0.5, 1, 1.5, 2]
const COLLECT_EVERY = [1, 2, 5, 10] // giây giữa hai lần chụp khi thu thập dữ liệu
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
// tên camera trên Linux/Windows hay kèm mã USB "(1234:abcd)", bỏ đi cho dễ đọc
const camName = (d) => (d.label || 'Camera').replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)$/i, '')

// Video hiện bằng thẻ <video> có nút điều khiển, khung nhận dạng vẽ trên canvas trong suốt đè lên.
// Vòng nhận dạng chạy độc lập với việc phát: đang phát thì gửi liên tục, tạm dừng thì chỉ gửi khung đang đứng.
export default function LiveView({ minConf, incoming, onTaken }) {
  const [source, setSource] = useState('file')
  const [videoFile, setVideoFile] = useState(null)
  const [camOn, setCamOn] = useState(false)
  const [camStarting, setCamStarting] = useState(false)
  const [cams, setCams] = useState([]) // các camera trình duyệt thấy, gồm cả điện thoại cắm USB ở chế độ webcam
  const [camId, setCamId] = useState(null)
  const { message } = AntApp.useApp()
  const [detect, setDetect] = useState(true)
  const [speed, setSpeed] = useState(1)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [fps, setFps] = useState(null)
  const [collecting, setCollecting] = useState(false)
  const [every, setEvery] = useState(2)
  const [collected, setCollected] = useState(null) // { saved, skipped, total, last }

  const video = useRef(null)
  const overlay = useRef(null)
  const stream = useRef(null)
  const camReq = useRef(0) // tăng mỗi lần tắt/đổi nguồn/rời tab để huỷ lần mở webcam đang chờ quyền
  const knownCams = useRef(new Set()) // để báo khi vừa cắm thêm camera
  const shown = useRef(null) // kết quả đang vẽ: { data, seek }
  const seekId = useRef(0) // tăng mỗi lần tua, kết quả của lần tua cũ bị bỏ
  const session = useRef(null) // mã phiên thu thập, mỗi lần bật là một phiên (notebook gom ảnh theo phiên)
  const conf = useRef(minConf)
  conf.current = minConf

  const objects = (result?.objects || []).filter((o) => passConf(o, minConf))
  const hasMedia = source === 'file' ? !!videoFile : camOn

  const clearResult = () => {
    shown.current = null
    setResult(null)
    setFps(null)
  }

  const stopCamera = () => {
    camReq.current += 1
    setCamStarting(false)
    stream.current?.getTracks().forEach((t) => t.stop())
    stream.current = null
    if (video.current?.srcObject) video.current.srcObject = null
    setCamOn(false)
    setReady(false)
  }

  const turnOffCamera = () => {
    stopCamera()
    clearResult()
  }

  useEffect(() => stopCamera, [])

  // danh sách camera; trước khi được cấp quyền trình duyệt giấu id và tên nên danh sách rỗng
  const refreshCams = async () => {
    const md = navigator.mediaDevices
    if (!md?.enumerateDevices) return
    const list = (await md.enumerateDevices()).filter((d) => d.kind === 'videoinput' && d.deviceId)
    const added = list.filter((d) => !knownCams.current.has(d.deviceId))
    if (knownCams.current.size && added.length) {
      message.info(`Đã nhận thêm camera: ${added.map(camName).join(', ')}`)
    }
    knownCams.current = new Set(list.map((d) => d.deviceId))
    setCams(list)
  }

  // cắm/rút điện thoại hoặc webcam USB thì cập nhật danh sách
  useEffect(() => {
    const md = navigator.mediaDevices
    if (!md?.addEventListener) return
    refreshCams()
    md.addEventListener('devicechange', refreshCams)
    return () => md.removeEventListener('devicechange', refreshCams)
  }, [])

  // video thả ở tab ảnh rồi chuyển sang đây
  useEffect(() => {
    if (!incoming) return
    setSource('file')
    setVideoFile({ file: incoming.file, id: incoming.id })
    onTaken?.()
  }, [incoming?.id])

  // gắn file video vào thẻ <video>; url được tạo và thu hồi cùng một chỗ
  useEffect(() => {
    const v = video.current
    if (source !== 'file' || !v) return
    setReady(false)
    setError(null)
    clearResult()
    if (!videoFile) {
      v.removeAttribute('src')
      v.load()
      return
    }
    const url = URL.createObjectURL(videoFile.file)
    v.srcObject = null
    v.src = url
    return () => URL.revokeObjectURL(url)
  }, [source, videoFile])

  useEffect(() => {
    if (video.current) {
      video.current.defaultPlaybackRate = speed
      video.current.playbackRate = speed
    }
  }, [speed, ready])

  // vẽ lớp khung theo từng frame màn hình
  useEffect(() => {
    let id
    const tick = () => {
      const v = video.current
      const c = overlay.current
      if (v && c && v.videoWidth) {
        if (c.width !== v.videoWidth) c.width = v.videoWidth
        if (c.height !== v.videoHeight) c.height = v.videoHeight
        const ctx = c.getContext('2d')
        ctx.clearRect(0, 0, c.width, c.height)
        const s = shown.current
        if (s && s.seek === seekId.current) {
          drawBoxes(ctx, s.data.objects.filter((o) => o.confidence >= conf.current), c.width, c.height)
        }
      }
      id = requestAnimationFrame(tick)
    }
    id = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(id)
  }, [])

  // vòng nhận dạng; mỗi lần bật lại / đổi nguồn là một vòng mới, vòng cũ tự thoát
  useEffect(() => {
    if (!detect || !ready || !hasMedia) return
    let alive = true
    const ctrl = new AbortController()
    const times = []
    let lastKey = null

    const run = async () => {
      while (alive) {
        const v = video.current
        if (!v || !v.videoWidth || v.readyState < 2) {
          await sleep(100)
          continue
        }
        const key = `${seekId.current}:${v.currentTime}`
        // tạm dừng và khung này đã nhận dạng rồi thì chờ
        if (v.paused && key === lastKey) {
          await sleep(100)
          continue
        }
        const sid = seekId.current
        const frame = await grab(v)
        let data
        try {
            data = await predict(frame, ctrl.signal, 'fast')
        } catch (err) {
          if (!alive || isCanceled(err)) return
          setError(errorText(err))
          setDetect(false)
          return
        }
        if (!alive) return
        lastKey = key
        if (sid !== seekId.current) continue // người dùng vừa tua, bỏ kết quả cũ
        shown.current = { data, seek: sid }
        setResult(data)
        times.push(performance.now())
        if (times.length > 10) times.shift()
        if (times.length > 1 && !v.paused) {
          setFps((((times.length - 1) * 1000) / (times[times.length - 1] - times[0])).toFixed(1))
        } else if (v.paused) {
          setFps(null)
        }
      }
    }
    run()
    return () => {
      alive = false
      ctrl.abort()
    }
  }, [detect, ready, hasMedia, source, videoFile])

  // thu thập dữ liệu train: cứ `every` giây gửi một khung gốc (không vẽ khung) cho server tự gán nhãn và lưu
  useEffect(() => {
    if (!collecting || !ready || !hasMedia) return
    let alive = true
    const ctrl = new AbortController()
    let lastKey = null

    const run = async () => {
      while (alive) {
        const v = video.current
        const key = `${seekId.current}:${v?.currentTime}`
        // video đang dừng ở khung đã gửi thì không gửi lại
        if (v && v.videoWidth && v.readyState >= 2 && !(v.paused && key === lastKey)) {
          lastKey = key
          let r
          try {
            r = await collectFrame(await grab(v), session.current, ctrl.signal)
          } catch (err) {
            if (!alive || isCanceled(err)) return
            setError(errorText(err))
            setCollecting(false)
            return
          }
          if (!alive) return
          setCollected((c) => ({
            saved: c.saved + (r.saved ? 1 : 0),
            skipped: c.skipped + (r.saved ? 0 : 1),
            total: c.total + (r.saved ? 1 : 0),
            last: r.saved ? `đã lưu: ${r.labels.map(labelText).join(', ')}` : `bỏ qua: ${r.reason}`,
          }))
        }
        await sleep(every * 1000)
      }
    }
    run()
    return () => {
      alive = false
      ctrl.abort()
    }
  }, [collecting, every, ready, hasMedia, source, videoFile])

  const toggleCollect = async (on) => {
    setError(null)
    if (!on) {
      setCollecting(false)
      return
    }
    let total = 0
    try {
      total = (await collectStats()).images
    } catch (err) {
      setError(errorText(err))
      return
    }
    const d = new Date()
    const p = (n) => String(n).padStart(2, '0')
    session.current = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
    setCollected({ saved: 0, skipped: 0, total, last: null })
    setCollecting(true)
  }

  const onSeeking = () => {
    seekId.current += 1
  }

  const onVideoError = () => {
    if (source !== 'file' || !videoFile) return
    setReady(false)
    setError(video.current?.error?.code === 4
      ? 'Trình duyệt không đọc được file này. File có thể bị hỏng hoặc dùng định dạng không hỗ trợ, hãy thử file MP4 khác.'
      : 'Không phát được video, thử chọn lại file.')
  }

  const pickVideo = (f) => {
    if (!f.type.startsWith('video/')) {
      setError('Chỉ nhận file video, ví dụ .mp4 hoặc .webm')
    } else {
      setVideoFile({ file: f, id: Date.now() })
    }
    return Upload.LIST_IGNORE
  }

  const removeVideo = () => {
    video.current?.pause()
    setVideoFile(null)
  }

  const startCamera = async (deviceId = camId) => {
    setError(null)
    clearResult()
    // trình duyệt chỉ cho dùng camera trên localhost hoặc https
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Trình duyệt không cho dùng webcam ở địa chỉ này. Hãy mở trang bằng http://localhost hoặc https.')
      return
    }
    const id = ++camReq.current
    setCamStarting(true)
    let s = null
    try {
      s = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
        audio: false,
      })
      // trong lúc chờ quyền người dùng đã đổi nguồn, tắt hoặc rời tab: đóng luôn camera vừa mở
      if (id !== camReq.current) {
        s.getTracks().forEach((t) => t.stop())
        return
      }
      stream.current = s
      // đã có quyền: lấy được tên các camera, đánh dấu camera đang dùng
      setCamId(s.getVideoTracks()[0]?.getSettings().deviceId || deviceId)
      refreshCams()
      // rút webcam hoặc thu hồi quyền khi đang chạy
      s.getVideoTracks()[0]?.addEventListener('ended', () => {
        if (stream.current !== s) return
        stopCamera()
        clearResult()
        setError('Webcam đã ngắt kết nối.')
      })
      const v = video.current
      v.removeAttribute('src')
      v.srcObject = s
      await v.play()
      if (id !== camReq.current) return
      setCamOn(true)
    } catch (err) {
      if (id !== camReq.current) {
        s?.getTracks().forEach((t) => t.stop())
        return
      }
      stopCamera()
      setError(err?.name === 'NotAllowedError'
        ? 'Trình duyệt chưa được cấp quyền dùng camera. Bấm biểu tượng camera trên thanh địa chỉ để cho phép rồi thử lại.'
        : err?.name === 'NotFoundError' || err?.name === 'OverconstrainedError'
          ? (deviceId ? 'Không mở được camera đã chọn, có thể đã bị rút ra. Chọn camera khác trong danh sách.'
            : 'Không tìm thấy webcam nào trên máy.')
          : err?.name === 'NotReadableError'
            ? 'Webcam đang được ứng dụng khác sử dụng.'
            : 'Không mở được webcam. Kiểm tra trình duyệt đã được cấp quyền dùng camera.')
    } finally {
      if (id === camReq.current) setCamStarting(false)
    }
  }

  // đổi camera khi đang bật thì mở lại ngay bằng camera mới
  const pickCam = (id) => {
    setCamId(id)
    if (camOn || camStarting) {
      stopCamera()
      clearResult()
      startCamera(id)
    }
  }

  const changeSource = (s) => {
    video.current?.pause()
    stopCamera()
    setReady(false)
    setError(null)
    clearResult()
    if (s === 'camera' && video.current) {
      video.current.removeAttribute('src')
      video.current.load()
    }
    setSource(s)
  }

  const saveFrame = () => {
    const v = video.current
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    const ctx = c.getContext('2d')
    ctx.drawImage(v, 0, 0)
    const s = shown.current
    if (s && s.seek === seekId.current) drawBoxes(ctx, objects, c.width, c.height)
    const base = videoFile ? videoFile.file.name.replace(/\.[^.]+$/, '') : 'webcam'
    const at = source === 'file' ? `_${v.currentTime.toFixed(1)}s` : ''
    downloadCanvas(c, `${base}${at}_ketqua.jpg`)
  }

  const c = count(objects)

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} md={14} xl={16}>
        <Card title="Nguồn hình" size="small"
          extra={<Segmented size="small" value={source} onChange={changeSource}
            options={[{ label: 'File video', value: 'file' }, { label: 'Webcam', value: 'camera' }]} />}>
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            {source === 'file' && (
              <Upload.Dragger accept="video/*" showUploadList={false} beforeUpload={pickVideo}
                className="video-drop">
                <p className="ant-upload-text">
                  {videoFile ? videoFile.file.name : 'Thả video vào đây hoặc bấm để chọn'}
                </p>
                <p className="ant-upload-hint">
                  {videoFile ? 'Thả hoặc chọn video khác để đổi' : 'MP4 hoặc WebM'}
                </p>
              </Upload.Dragger>
            )}

            {/* .frame ôm sát kích thước video để canvas đè lên khớp từng pixel */}
            <div className={'stage' + (hasMedia ? '' : ' hidden')}>
              <div className="frame">
                <video ref={video} muted playsInline controls={source === 'file'}
                  onLoadedData={() => setReady(true)} onSeeking={onSeeking} onError={onVideoError}
                  onRateChange={(e) => setSpeed(e.currentTarget.playbackRate)} />
                <canvas ref={overlay} />
              </div>
            </div>
            {!hasMedia && (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
                description={source === 'camera'
                  ? 'Bấm "Bật webcam" và cho phép trình duyệt dùng camera.'
                  : 'Chọn một file video để bắt đầu.'} />
            )}

            <div className="live-bar">
              <Space wrap>
                {source === 'file' ? (
                  <Button onClick={removeVideo} disabled={!videoFile}>Bỏ video</Button>
                ) : camOn ? (
                  <Button danger onClick={turnOffCamera}>Tắt webcam</Button>
                ) : (
                  <Button type="primary" onClick={() => startCamera()} loading={camStarting}>
                    {camStarting ? 'Đang mở webcam…' : 'Bật webcam'}
                  </Button>
                )}
                {source === 'camera' && cams.length > 0 && (
                  <Select className="cam-select" value={cams.some((d) => d.deviceId === camId) ? camId : undefined}
                    placeholder="Chọn camera" onChange={pickCam} popupMatchSelectWidth={false}
                    aria-label="Chọn camera"
                    options={cams.map((d) => ({ value: d.deviceId, label: camName(d) }))} />
                )}
                <Button onClick={saveFrame} disabled={!ready || !hasMedia}>Lưu khung hình</Button>
              </Space>
              <Space wrap>
                {source === 'file' && (
                  <span className="live-opt">
                    Tốc độ
                    <Segmented size="small" value={speed} onChange={setSpeed}
                      options={SPEEDS.map((s) => ({ label: `${s}x`, value: s }))} />
                  </span>
                )}
                <span className="live-opt">
                  Thu thập dữ liệu
                  <Switch size="small" checked={collecting} onChange={toggleCollect} disabled={!hasMedia} />
                  {collecting && (
                    <Select size="small" value={every} onChange={setEvery} aria-label="Chụp mỗi"
                      options={COLLECT_EVERY.map((s) => ({ value: s, label: `mỗi ${s} giây` }))} />
                  )}
                </span>
                <span className="live-opt">
                  Nhận dạng
                  <Switch size="small" checked={detect} onChange={(on) => { setDetect(on); setError(null) }} />
                </span>
              </Space>
            </div>
            {collected && (
              <Alert type={collecting ? 'info' : 'success'} showIcon
                message={`${collecting ? 'Đang thu thập' : 'Đã dừng thu thập'}: phiên này lưu ${collected.saved} ảnh, bỏ ${collected.skipped} ảnh. Tổng ${collected.total} ảnh chờ train.`}
                description={(collected.last ? `Khung gần nhất ${collected.last}. ` : '')
                  + 'Chỉ lưu khung mà hai model cùng đồng ý và chắc chắn tình trạng; lần train sau notebook tự dùng.'} />
            )}
            {source === 'camera' && (
              <p className="cam-hint">
                Dùng camera điện thoại: cắm cáp USB vào máy, trên điện thoại chọn chế độ USB là "Webcam"
                (Android 14 trở lên) hoặc mở app webcam như DroidCam, Iriun. Camera mới sẽ tự hiện trong danh sách
                chọn camera sau khi đã bật webcam một lần.
              </p>
            )}
          </Space>
        </Card>
      </Col>

      <Col xs={24} md={10} xl={8}>
        <Card title="Khung hình đang xem" size="small">
          {error && <Alert type="error" message={error} showIcon className="mb16" />}
          {result ? (
            <>
              <SummaryBar c={c}
                note={fps ? `${fps} khung/giây, mỗi khung ${result.inference_ms} ms` : `Xử lý ${result.inference_ms} ms`} />
              <ResultTable objects={objects} />
            </>
          ) : (
            !error && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={!hasMedia ? 'Kết quả của khung hình đang xem sẽ hiện ở đây.'
                : detect ? 'Đang nhận dạng khung hình…' : 'Bật "Nhận dạng" để xem kết quả.'} />
          )}
        </Card>
      </Col>
    </Row>
  )
}

// 'banana_rotten' -> 'Chuối hỏng'
function labelText(name) {
  const [f, st] = name.split('_')
  return `${FRUIT[f] || f} ${(STATE[st] || st).toLowerCase()}`
}

function grab(v) {
  return new Promise((resolve) => {
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    c.getContext('2d').drawImage(v, 0, 0)
    c.toBlob((b) => resolve(new File([b], 'frame.jpg', { type: 'image/jpeg' })), 'image/jpeg', 0.85)
  })
}
