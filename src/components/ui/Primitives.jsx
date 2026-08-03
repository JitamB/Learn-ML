/**
 * Note — Callout / Alert box
 * color: 'info' | 'success' | 'warning' | 'danger' | 'purple'
 * icon:  any Tabler icon class name (e.g. 'ti-info-circle')
 */
export function Note({ color = 'info', icon = 'ti-info-circle', children }) {
  return (
    <div className={`note ${color}`}>
      <i className={`ti ${icon} note-icon`} aria-hidden="true" />
      <div>{children}</div>
    </div>
  );
}

/**
 * Table — Styled data table
 * heads: string[]
 * rows:  (string | ReactNode)[][]
 */
export function Table({ heads, rows }) {
  return (
    <div className="ml-table-wrap">
      <table className="ml-table">
        <thead>
          <tr>
            {heads.map(h => <th key={h}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => <td key={j}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Grid — Responsive multi-column layout
 * cols: 1 | 2 | 3
 * gap:  number (px)
 */
export function Grid({ cols = 2, gap = 10, children }) {
  return (
    <div
      className={`ml-grid ml-grid-${cols}`}
      style={{ gap }}
    >
      {children}
    </div>
  );
}

/**
 * Card — Coloured info card
 * color: 'info' | 'success' | 'warning' | 'danger' | 'purple'
 * title: optional heading string
 */
export function Card({ color = 'info', title, children }) {
  return (
    <div className={`info-card ${color}`}>
      {title && <div className="info-card-title">{title}</div>}
      <div className="info-card-body">{children}</div>
    </div>
  );
}

/**
 * VizBox — Wrapper box for interactive canvas visualizations
 */
export function VizBox({ children }) {
  return <div className="viz-box">{children}</div>;
}

/**
 * SliderRow — Labelled range input with live value display
 */
export function SliderRow({ label, min, max, step, value, onChange, fmt = v => v.toFixed(2) }) {
  return (
    <div className="slider-row">
      <span className="slider-row-label">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(+e.target.value)}
      />
      <span className="slider-row-value">{fmt(value)}</span>
    </div>
  );
}

/**
 * Badge — Coloured pill label
 */
export function Badge({ color = 'info', children }) {
  return <span className={`badge ${color}`}>{children}</span>;
}
