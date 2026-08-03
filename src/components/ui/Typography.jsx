/** Mx — Math display block or inline */
export function Mx({ block, children }) {
  if (block) {
    return <span className="math-block">{children}</span>;
  }
  return <code className="math-inline">{children}</code>;
}

/** H2 — Section heading */
export function H2({ c, children }) {
  return <h2 className="h2">{c ?? children}</h2>;
}

/** H3 — Sub-section heading */
export function H3({ c, children }) {
  return <h3 className="h3">{c ?? children}</h3>;
}

/** P — Body paragraph */
export function P({ c, s, children }) {
  return <p className="p" style={s}>{c ?? children}</p>;
}
