import { useState } from 'react';

/**
 * QA — Collapsible Q&A accordion item
 * Props: q (question string), a (answer — may contain HTML markup)
 */
export function QA({ q, a }) {
  const [open, setOpen] = useState(false);
  const id = `qa-${q.slice(0, 20).replace(/\s+/g, '-').toLowerCase()}`;

  return (
    <div className="qa-item">
      <button
        className="qa-trigger"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
      >
        <span className="qa-q-text">Q: {q}</span>
        <i className="ti ti-chevron-down qa-chevron" aria-hidden="true" />
      </button>
      {open && (
        <div
          id={id}
          className="qa-body"
          dangerouslySetInnerHTML={{ __html: a }}
        />
      )}
    </div>
  );
}

export default QA;
