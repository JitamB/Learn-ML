import { useState } from 'react';
import { highlightLine } from './highlight.jsx';

/**
 * Polished code block with:
 * - macOS-style header (coloured dots + language pill + copy button)
 * - VS Code Dark+ styled syntax highlighting
 * - Monospace body with horizontal scroll
 */
export function Code({ children, lang = 'python' }) {
  const [copied, setCopied] = useState(false);
  const text = (typeof children === 'string' ? children : String(children)).trim();
  const lines = text.split('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };

  return (
    <div className="code-block">
      <div className="code-block-header">
        <div className="code-block-dots">
          <span className="code-block-dot red" />
          <span className="code-block-dot yellow" />
          <span className="code-block-dot green" />
        </div>
        <span className="code-block-lang">{lang}</span>
        <button className={`code-block-copy${copied ? ' copied' : ''}`} onClick={handleCopy}>
          <i className={`ti ${copied ? 'ti-check' : 'ti-copy'}`} style={{ fontSize: 11 }} />
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <div className="code-block-body">
        <pre>{lines.map((line, i) => <div key={i}>{highlightLine(line)}</div>)}</pre>
      </div>
    </div>
  );
}

export default Code;
