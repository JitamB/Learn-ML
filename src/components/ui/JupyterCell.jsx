import { useState } from 'react';
import { highlightLine } from './highlight.jsx';

export function Cell({ type = "code", output, children, num }) {
  const [collapsed, setCollapsed] = useState(false);

  if (type === "code") {
    const lines = children.trim().split("\n");
    return (
      <div style={{ display: "flex", gap: 0, marginBottom: 4, fontFamily: "var(--font-mono)" }}>
        <div style={{ width: 65, flexShrink: 0, paddingTop: 10, paddingRight: 10, textAlign: "right", fontSize: 11.5, color: "#569CD6", userSelect: "none", lineHeight: 1.6 }}>
          In [{num}]:
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ background: "var(--code-bg)", borderRadius: "var(--border-radius-md)", border: "1px solid var(--code-border)", position: "relative" }}>
            <button onClick={() => setCollapsed(!collapsed)} style={{ position: "absolute", top: 6, right: 8, background: "transparent", border: "none", cursor: "pointer", color: "var(--color-text-tertiary)", fontSize: 11, padding: "2px 6px", borderRadius: 3, zIndex: 1 }}>
              {collapsed ? "▶" : "▼"}
            </button>
            {!collapsed && (
              <pre style={{ margin: 0, padding: "10px 36px 10px 14px", fontSize: 12.5, lineHeight: 1.65, overflowX: "auto", color: "var(--code-text)", whiteSpace: "pre" }}>
                {lines.map((line, i) => (
                  <div key={i}>{highlightLine(line)}</div>
                ))}
              </pre>
            )}
            {collapsed && (
              <div style={{ padding: "6px 36px 6px 14px", fontSize: 11.5, color: "var(--color-text-tertiary)" }}>
                {lines.length} lines hidden
              </div>
            )}
          </div>
          {output && !collapsed && (
            <div style={{ display: "flex", gap: 0 }}>
              <div style={{ width: 0, flexShrink: 0 }} />
              <div style={{ flex: 1, borderLeft: "3px solid var(--code-border)", marginLeft: 2, paddingLeft: 12, paddingTop: 6, paddingBottom: 6, marginBottom: 4 }}>
                <pre style={{ margin: 0, fontSize: 12, lineHeight: 1.6, color: "var(--color-text-secondary)", whiteSpace: "pre-wrap", fontFamily: "var(--font-mono)" }}>{output}</pre>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", gap: 0, marginBottom: 4 }}>
      <div style={{ width: 65, flexShrink: 0 }} />
      <div style={{ flex: 1, padding: "6px 4px 6px 0", fontSize: 13.5, lineHeight: 1.75, color: "var(--color-text-secondary)" }}
        dangerouslySetInnerHTML={{ __html: children }} />
    </div>
  );
}

export const MDCell = ({ children }) => <Cell type="md">{children}</Cell>;

export const SectionHeader = ({ n, title, color = "info" }) => (
  <div style={{ display: "flex", gap: 0, marginBottom: 8, marginTop: 20 }}>
    <div style={{ width: 65, flexShrink: 0 }} />
    <div style={{ flex: 1, background: `var(--color-background-${color})`, border: `1px solid var(--color-border-${color})`, borderRadius: "var(--border-radius-md)", padding: "8px 14px", display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontSize: 14, fontWeight: 500, color: `var(--color-text-${color})` }}>{title}</span>
    </div>
  </div>
);
