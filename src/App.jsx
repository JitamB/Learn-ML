import { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useParams, useLocation } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar.jsx';
import { MODULES, getModuleByPage, getCategoryForModule } from './data/nav.js';
import { Helmet } from 'react-helmet-async';

/* ── Lazy page imports ──────────────────────────────────────── */
import { lazy, Suspense } from 'react';

const PAGE_MAP = {
  MLFundamentals: lazy(() => import('./pages/MLFundamentals.jsx')),
};

/* ── Loading fallback ───────────────────────────────────────── */
function PageLoader() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '60vh', gap: 10, color: 'var(--color-text-tertiary)', fontSize: 13,
    }}>
      <i className="ti ti-loader-2" style={{ fontSize: 20, animation: 'spin 1s linear infinite' }} />
      Loading module…
    </div>
  );
}

/* ── Module route wrapper ───────────────────────────────────── */
function ModulePage() {
  const { page } = useParams();
  const PageComponent = PAGE_MAP[page];
  const mod = getModuleByPage(page);

  if (!PageComponent) {
    return (
      <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--color-text-tertiary)' }}>
        <i className="ti ti-mood-confused" style={{ fontSize: 40, display: 'block', marginBottom: 12 }} />
        Module {page} not found.
      </div>
    );
  }

  return (
    <Suspense fallback={<PageLoader />}>
      {mod && (
        <Helmet>
          <title>{mod.title} | Machine Learning Guide</title>
          <meta name="description" content={`Learn about ${mod.title} in this comprehensive Machine Learning module.`} />
          <meta property="og:title" content={`${mod.title} | Machine Learning Guide`} />
          <meta property="twitter:title" content={`${mod.title} | Machine Learning Guide`} />
        </Helmet>
      )}
      <PageComponent />
    </Suspense>
  );
}

/* ── Topbar ─────────────────────────────────────────────────── */
function TopBar({ onToggleSidebar, theme, onToggleTheme }) {
  const location = useLocation();
  const pagePart = location.pathname.split('/module/')[1];
  const mod = getModuleByPage(pagePart);
  const category = mod ? getCategoryForModule(mod.id) : '';

  return (
    <header className="topbar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <button className="topbar-toggle" onClick={onToggleSidebar} aria-label="Toggle sidebar">
          <i className="ti ti-layout-sidebar-left-expand" aria-hidden="true" />
        </button>
        <div className="topbar-breadcrumb">
          {category && (
            <>
              <span className="topbar-breadcrumb-category">{category}</span>
              <i className="ti ti-chevron-right topbar-breadcrumb-sep" aria-hidden="true" />
            </>
          )}
          <span className="topbar-breadcrumb-title">{mod?.title ?? 'Machine Learning'}</span>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {mod && (
          <span className="topbar-progress">
            {mod.id} / {MODULES.length}
          </span>
        )}
        <button onClick={onToggleTheme} style={{ background:'transparent', border:'none', color:'var(--color-text-secondary)', cursor:'pointer', fontSize:20, display:'flex', alignItems:'center' }} aria-label="Toggle theme">
          {theme === 'dark' ? <i className="ti ti-sun" /> : <i className="ti ti-moon" />}
        </button>
      </div>
    </header>
  );
}

/* ── Content shell (topbar + scrollable page) ───────────────── */
function ContentShell({ onToggleSidebar, theme, onToggleTheme }) {
  return (
    <div className="content-shell">
      <TopBar onToggleSidebar={onToggleSidebar} theme={theme} onToggleTheme={onToggleTheme} />
      <div className="content-scroll">
        <div className="content-page">
          <Routes>
            <Route path="/" element={<Navigate to={`/module/${MODULES[0]?.page}`} replace />} />
            <Route path="/module/:page" element={<ModulePage />} />
            <Route path="*" element={<Navigate to={`/module/${MODULES[0]?.page}`} replace />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}

/* ── Root App ───────────────────────────────────────────────── */
export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const location = useLocation();

  // Theme state
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light');

  // On mobile, collapse sidebar on route change
  useEffect(() => {
    if (window.innerWidth <= 768) {
      setSidebarCollapsed(true);
    }
  }, [location.pathname]);

  // Initialise to collapsed on small screens
  useEffect(() => {
    setSidebarCollapsed(window.innerWidth <= 768);
  }, []);

  const toggle = () => setSidebarCollapsed(p => !p);
  const close  = () => setSidebarCollapsed(true);

  return (
    <div className="app-shell">
      <Sidebar collapsed={sidebarCollapsed} onClose={close} />
      <ContentShell onToggleSidebar={toggle} theme={theme} onToggleTheme={toggleTheme} />

      {/* Spin keyframe for loading icon */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
