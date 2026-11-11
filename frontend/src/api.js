import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 60000,
})

export async function predict(file, signal, mode = 'careful') {
  const form = new FormData()
  form.append('file', file)
  form.append('mode', mode)
  const res = await api.post('/api/predict', form, { signal })
  return res.data
}

export const isCanceled = (err) => axios.isCancel(err)

export function errorText(err) {
  if (err.response?.data?.detail) return err.response.data.detail
  if (err.code === 'ECONNABORTED') return 'Server phản hồi quá lâu'
  if (!err.response) return 'Không kết nối được tới server'
  return 'Có lỗi xảy ra, thử lại sau'
}

export async function collectFrame(file, session, signal) {
  const form = new FormData()
  form.append('file', file)
  form.append('session', session)
  const res = await api.post('/api/collect', form, { signal })
  return res.data
}

export async function collectStats() {
  const res = await api.get('/api/collect/stats')
  return res.data
}
