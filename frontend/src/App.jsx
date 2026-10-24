import { useEffect, useState } from 'react'
import { App as AntApp, Grid, Layout, Slider, Tabs } from 'antd'
import ImagesView from './ImagesView'
import LiveView from './LiveView'
import { FruitSticker } from './Sticker'
import { pct } from './labels'

const { Content } = Layout

export default function App() {
  const [minConf, setMinConf] = useState(0.5)
  const [tab, setTab] = useState('images')
  const [video, setVideo] = useState(null)
  const screens = Grid.useBreakpoint()
  const { message } = AntApp.useApp()

  // thả file ra ngoài vùng tải lên thì Chrome sẽ mở file và rời trang, chặn lại
  useEffect(() => {
    const stop = (e) => e.preventDefault()
    window.addEventListener('dragover', stop)
    window.addEventListener('drop', stop)
    return () => {
      window.removeEventListener('dragover', stop)
      window.removeEventListener('drop', stop)
    }
  }, [])

  // video thả nhầm vào tab ảnh thì chuyển sang tab video
  const openVideo = (f) => {
    setVideo({ file: f, id: Date.now() })
    setTab('live')
    message.info(`Đã chuyển sang tab "Camera và video" để xử lý ${f.name}`)
  }

  const threshold = (
    <label className="threshold">
      <span>Ngưỡng tin cậy <b>{pct(minConf)}%</b></span>
      <Slider min={0.25} max={0.95} step={0.05} value={minConf} onChange={setMinConf}
        tooltip={{ open: false }} />
    </label>
  )

  const tabs = [
    { key: 'images', label: 'Ảnh', children: <ImagesView minConf={minConf} onVideo={openVideo} /> },
    {
      key: 'live',
      label: 'Camera và video',
      children: <LiveView minConf={minConf} incoming={video} onTaken={() => setVideo(null)} />,
      destroyOnHidden: true,
    },
  ]

  return (
    <Layout className="page">
      <header className="top">
        <div className="wrap top-inner">
          <div>
            <h1>Kiểm tra độ tươi trái cây</h1>
          </div>
          <div className="top-fruits" aria-label="Các loại quả nhận diện được">
            <FruitSticker label="apple" />
            <FruitSticker label="banana" />
            <FruitSticker label="orange" />
          </div>
        </div>
      </header>

      <Content className="wrap content">
        {/* màn hẹp: đặt thanh ngưỡng trên thanh tab để tab không bị gom vào "..." */}
        {!screens.md && threshold}
        <Tabs activeKey={tab} onChange={setTab} items={tabs}
          tabBarExtraContent={screens.md ? threshold : null} />
      </Content>
    </Layout>
  )
}
