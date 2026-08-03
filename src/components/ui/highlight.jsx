const KEYWORDS = ["import", "from", "as", "def", "class", "return", "for", "in", "if", "else", "elif", "while", "with", "try", "except", "pass", "lambda", "and", "or", "not", "True", "False", "None", "yield", "raise", "assert", "break", "continue", "is", "global", "nonlocal"];
const BUILTINS = ["print", "len", "range", "list", "dict", "set", "tuple", "int", "float", "str", "bool", "type", "zip", "enumerate", "map", "filter", "sorted", "sum", "min", "max", "round", "abs", "open", "super", "self"];

/**
 * Tokenizes one line of Python into VS Code Dark+ styled spans — keywords
 * blue, strings orange, numbers green, builtins yellow, capitalized
 * identifiers teal, comments green. Shared by Code.jsx and JupyterCell.jsx
 * so both render code in the same theme.
 */
export function highlightLine(line) {
  if (line.trimStart().startsWith("#")) {
    return <span style={{ color: "#6A9955" }}>{line}</span>;
  }
  const parts = [];
  let rest = line;
  let key = 0;

  while (rest.length > 0) {
    const strMatch = rest.match(/^("""[\s\S]*?"""|'''[\s\S]*?'''|"[^"\n]*"|'[^'\n]*')/);
    if (strMatch && strMatch.index === 0) {
      parts.push(<span key={key++} style={{ color: "#CE9178" }}>{strMatch[0]}</span>);
      rest = rest.slice(strMatch[0].length);
      continue;
    }
    const numMatch = rest.match(/^(\b\d+\.?\d*\b)/);
    if (numMatch) {
      parts.push(<span key={key++} style={{ color: "#B5CEA8" }}>{numMatch[0]}</span>);
      rest = rest.slice(numMatch[0].length);
      continue;
    }
    const wordMatch = rest.match(/^([a-zA-Z_][a-zA-Z0-9_]*)/);
    if (wordMatch) {
      const w = wordMatch[0];
      let color = "inherit";
      if (KEYWORDS.includes(w)) color = "#569CD6";
      else if (BUILTINS.includes(w)) color = "#DCDCAA";
      else if (/^[A-Z]/.test(w)) color = "#4EC9B0";
      parts.push(<span key={key++} style={{ color }}>{w}</span>);
      rest = rest.slice(w.length);
      continue;
    }
    if (rest[0] === "#") {
      parts.push(<span key={key++} style={{ color: "#6A9955" }}>{rest}</span>);
      break;
    }
    const opMatch = rest.match(/^([=<>!+\-*\/\[\]{}(),.:@%&|^~;]+)/);
    if (opMatch) {
      parts.push(<span key={key++} style={{ color: "#D4D4D4" }}>{opMatch[0]}</span>);
      rest = rest.slice(opMatch[0].length);
      continue;
    }
    parts.push(<span key={key++}>{rest[0]}</span>);
    rest = rest.slice(1);
  }
  return parts;
}
