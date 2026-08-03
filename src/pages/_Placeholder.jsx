/**
 * Placeholder page generator — used for modules not yet written.
 * Each module has its own stub that imports this component.
 */
import NavButtons from '../components/layout/NavButtons.jsx';

export default function PlaceholderPage({ moduleId, title, subtitle }) {
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module {String(moduleId).padStart(2, '0')}</div>
        <h1 className="page-header-title">{title}</h1>
        <p className="page-header-subtitle">{subtitle}</p>
      </div>

      <div style={{
        display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', padding: '3rem 1rem', gap: 16,
        border: '1px dashed var(--color-border-tertiary)',
        borderRadius: 'var(--border-radius-lg)',
        color: 'var(--color-text-tertiary)',
        background: 'var(--color-background-secondary)',
        marginBottom: '2rem',
      }}>
        <i className="ti ti-tools" style={{ fontSize: 36 }} aria-hidden="true" />
        <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--color-text-secondary)' }}>
          Content coming soon
        </div>
        <div style={{ fontSize: 13, textAlign: 'center', maxWidth: 380 }}>
          This module hasn't been written yet. Check back after the next content batch.
        </div>
      </div>

      <NavButtons moduleId={moduleId} />
    </div>
  );
}
