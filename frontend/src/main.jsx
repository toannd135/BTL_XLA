import React from 'react'
import ReactDOM from 'react-dom/client'
import { App as AntApp, ConfigProvider } from 'antd'
import viVN from 'antd/locale/vi_VN'
import App from './App'
import './App.css'

const theme = {
  token: {
    fontFamily: "'Be Vietnam Pro', system-ui, sans-serif",
    colorPrimary: '#2e6b3a',
    colorText: '#1e2a1f',
    colorTextSecondary: '#5b6b5c',
    colorBgLayout: '#eef3ea',
    colorBorder: '#cfdac9',
    colorBorderSecondary: '#e1e8dc',
    colorError: '#b3412f',
    borderRadius: 8,
    borderRadiusLG: 12,
    boxShadow: 'none',
    boxShadowSecondary: '0 4px 16px rgba(29, 66, 39, 0.12)',
  },
  components: {
    Card: { headerFontSize: 15 },
    Table: { headerBg: '#f4f8f1', headerColor: '#5b6b5c' },
    Tabs: { titleFontSize: 15, inkBarColor: '#f2c230' },
    Slider: {
      trackBg: '#2e6b3a', trackHoverBg: '#1d4227',
      handleColor: '#2e6b3a', handleActiveColor: '#1d4227', railBg: '#d3ddcd',
    },
  },
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ConfigProvider theme={theme} locale={viVN}>
      {/* AntApp cho message dùng đúng theme của ConfigProvider */}
      <AntApp>
        <App />
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>,
)
