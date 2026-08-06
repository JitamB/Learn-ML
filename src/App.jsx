import { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useParams, useLocation } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar.jsx';
import Home from './pages/Home.jsx';
import { MODULES, getModuleByPage, getCategoryForModule } from './data/nav.js';
import { Helmet } from 'react-helmet-async';

/* ── Lazy page imports ──────────────────────────────────────── */
import { lazy, Suspense } from 'react';

const PAGE_MAP = {
  MLFundamentals: lazy(() => import('./pages/MLFundamentals.jsx')),
  EDA: lazy(() => import('./pages/EDA.jsx')),
  DataCleaningFeatureEngineering: lazy(() => import('./pages/DataCleaningFeatureEngineering.jsx')),
  LinearModels: lazy(() => import('./pages/LinearModels.jsx')),
  KNNDecisionTrees: lazy(() => import('./pages/KNNDecisionTrees.jsx')),
  SVMNaiveBayes: lazy(() => import('./pages/SVMNaiveBayes.jsx')),
  BaggingRandomForest: lazy(() => import('./pages/BaggingRandomForest.jsx')),
  Boosting: lazy(() => import('./pages/Boosting.jsx')),
  Clustering: lazy(() => import('./pages/Clustering.jsx')),
  DimensionalityReduction: lazy(() => import('./pages/DimensionalityReduction.jsx')),
  AssociationRulesAnomalyDetection: lazy(() => import('./pages/AssociationRulesAnomalyDetection.jsx')),
  RecommenderSystems: lazy(() => import('./pages/RecommenderSystems.jsx')),
  SemiSupervisedLearning: lazy(() => import('./pages/SemiSupervisedLearning.jsx')),
  SelfSupervisedLearning: lazy(() => import('./pages/SelfSupervisedLearning.jsx')),
  ValidationBiasVariance: lazy(() => import('./pages/ValidationBiasVariance.jsx')),
  Metrics: lazy(() => import('./pages/Metrics.jsx')),
  Regularization: lazy(() => import('./pages/Regularization.jsx')),
  HyperparameterTuning: lazy(() => import('./pages/HyperparameterTuning.jsx')),
  TimeSeriesAnalysis: lazy(() => import('./pages/TimeSeriesAnalysis.jsx')),
  InterpretabilityFairness: lazy(() => import('./pages/InterpretabilityFairness.jsx')),
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
  const pagePart = location.pathname.replace(/^\//, '');
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
        {/* <button onClick={onToggleTheme} style={{ background:'transparent', border:'none', color:'var(--color-text-secondary)', cursor:'pointer', fontSize:20, display:'flex', alignItems:'center' }} aria-label="Toggle theme">
          {theme === 'dark' ? <i className="ti ti-sun" /> : <i className="ti ti-moon" />}
        </button> */}
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
            <Route path="/:page" element={<ModulePage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
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

  // Home is a standalone page — no sidebar, no topbar, just the page itself.
  if (location.pathname === '/') {
    return <Home />;
  }

  return (
    <div className="app-shell">
      <Sidebar collapsed={sidebarCollapsed} onClose={close} />
      <ContentShell onToggleSidebar={toggle} theme={theme} onToggleTheme={toggleTheme} />

      {/* Spin keyframe for loading icon */}
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
