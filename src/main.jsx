import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './styles/global.css'
import './styles/layout.css'
import './styles/components.css'
import { HelmetProvider } from 'react-helmet-async'

const basePath = import.meta.env.BASE_URL.replace(/\/$/, '')
const searchParams = new URLSearchParams(window.location.search)
const redirectedPath = searchParams.get('p')

if (redirectedPath) {
  window.history.replaceState(
    null,
    '',
    `${basePath}/${redirectedPath.replace(/^\//, '')}`
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <HelmetProvider>
      <BrowserRouter basename={basePath}>
        <App />
      </BrowserRouter>
    </HelmetProvider>
  </React.StrictMode>
)
