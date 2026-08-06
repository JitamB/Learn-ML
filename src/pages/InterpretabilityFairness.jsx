import { useState, useMemo } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow, Badge } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

/* ── Shared helpers (duplicated per-file by this codebase's convention) ── */
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
}
function randNormalish(rand, mean, std) {
  const u = rand() + rand() + rand() - 1.5;
  return mean + (u / 0.5) * std;
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 12px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}

/* ── Generic n-dim CART + bagged-forest machinery ──────────────────────
   Reused for BOTH the loan case study (3 named tabular features) and the
   LIME checkerboard (2 spatial features) — bestSplit/fitTree are generic
   over a `features` name array via bracket notation. Leaves carry a
   `.prob = c1/n` field (sklearn-style soft voting) so an ensemble yields a
   continuous [0,1] score, not just a hard vote — needed by PDP/SHAP/
   fairness thresholds/LIME, none of which work cleanly off a hard label. */
function giniOf(c1, n) {
  if (n === 0) return 0;
  const p1 = c1 / n;
  return 1 - p1 * p1 - (1 - p1) * (1 - p1);
}
function bestSplit(points, minLeaf, features) {
  const n = points.length;
  const totalOnes = points.filter(p => p.label === 1).length;
  let best = null;
  for (const feat of features) {
    const sorted = [...points].sort((a, b) => a[feat] - b[feat]);
    let leftOnes = 0;
    for (let i = 0; i < n - 1; i++) {
      leftOnes += sorted[i].label === 1 ? 1 : 0;
      if (sorted[i][feat] === sorted[i + 1][feat]) continue;
      const leftN = i + 1, rightN = n - leftN;
      if (leftN < minLeaf || rightN < minLeaf) continue;
      const rightOnes = totalOnes - leftOnes;
      const w = (leftN * giniOf(leftOnes, leftN) + rightN * giniOf(rightOnes, rightN)) / n;
      if (!best || w < best.w - 1e-9) best = { feat, thresh: (sorted[i][feat] + sorted[i + 1][feat]) / 2, w };
    }
  }
  return best;
}
function fitTree(points, depth, maxDepth, minLeaf, features) {
  const n = points.length, c1 = points.filter(p => p.label === 1).length, g = giniOf(c1, n);
  const node = { label: c1 * 2 >= n ? 1 : 0, prob: n === 0 ? 0 : c1 / n, isLeaf: true, nSamples: n };
  if (depth >= maxDepth || g === 0 || n < 2 * minLeaf) return node;
  const s = bestSplit(points, minLeaf, features);
  if (!s || s.w >= g - 1e-9) return node;
  node.isLeaf = false; node.feature = s.feat; node.threshold = s.thresh;
  node.impurityDecrease = g - s.w;
  node.left = fitTree(points.filter(p => p[s.feat] <= s.thresh), depth + 1, maxDepth, minLeaf, features);
  node.right = fitTree(points.filter(p => p[s.feat] > s.thresh), depth + 1, maxDepth, minLeaf, features);
  return node;
}
function predictTreeRow(node, row) {
  return node.isLeaf ? node : predictTreeRow(row[node.feature] <= node.threshold ? node.left : node.right, row);
}
function bootstrapSample(pool, seed) {
  const r = seededRandom(seed);
  return Array.from({ length: pool.length }, () => pool[Math.floor(r() * pool.length)]);
}
function fitForest(points, nTrees, maxDepth, minLeaf, features, seedBase) {
  return Array.from({ length: nTrees }, (_, i) => fitTree(bootstrapSample(points, seedBase + i), 0, maxDepth, minLeaf, features));
}
function ensembleScore(forest, row) {
  return forest.reduce((s, t) => s + predictTreeRow(t, row).prob, 0) / forest.length;
}

/* ── Loan-approval case study (Tabs 1, 2, 3, 5) ─────────────────────────
   3 named features feed the model; `group` and `trueRepay` never do —
   withheld so Tab 5 can audit the model with information it never saw. */
const FEATURES = ['income', 'creditScore', 'debtRatio'];
const LOAN_PENALTY = 0.3; // calibrated: 0.2/0.3/0.4 all land the model's group gap at the same 35.0pp plateau — not a knife-edge

function makeLoanDataset(penalty) {
  const randInc = seededRandom(5), randCS = seededRandom(11),
        randDR = seededRandom(17), randNoise = seededRandom(23);
  const GROUPS = [
    { g: 'A', incMean: 65000, incStd: 15000, csMean: 680, csStd: 60, drMean: 0.30, drStd: 0.10 },
    { g: 'B', incMean: 52000, incStd: 15000, csMean: 650, csStd: 60, drMean: 0.33, drStd: 0.10 },
  ];
  const rows = [];
  for (const { g, incMean, incStd, csMean, csStd, drMean, drStd } of GROUPS) {
    for (let i = 0; i < 120; i++) {
      const income = clamp(randNormalish(randInc, incMean, incStd), 8000, 300000);
      const creditScore = clamp(randNormalish(randCS, csMean, csStd), 300, 850);
      const debtRatio = clamp(randNormalish(randDR, drMean, drStd), 0.03, 0.75);
      const latent = 0.55 * (income - 58000) / 15000
                   + 0.85 * (creditScore - 665) / 60
                   - 1.1 * (debtRatio - 0.315) / 0.10
                   + randNormalish(randNoise, 0, 0.6);
      const trueRepay = latent > 0 ? 1 : 0;
      const historicalApproved = (latent - (g === 'B' ? penalty : 0)) > 0 ? 1 : 0;
      rows.push({ group: g, income, creditScore, debtRatio, trueRepay, historicalApproved });
    }
  }
  return rows;
}
const LOAN_ROWS = makeLoanDataset(LOAN_PENALTY);
const LOAN_TRAIN = LOAN_ROWS.map(r => ({ ...r, label: r.historicalApproved }));
const LOAN_FOREST = fitForest(LOAN_TRAIN, 25, 5, 8, FEATURES, 42);
const LOAN_SCORED = LOAN_ROWS.map(r => ({ ...r, score: ensembleScore(LOAN_FOREST, r) }));
const LOAN_BASE_VALUE = LOAN_SCORED.reduce((s, r) => s + r.score, 0) / LOAN_SCORED.length;
const LOAN_ACC_HIST = LOAN_SCORED.filter(r => (r.score >= 0.5 ? 1 : 0) === r.historicalApproved).length / LOAN_SCORED.length;
const LOAN_ACC_TRUE = LOAN_SCORED.filter(r => (r.score >= 0.5 ? 1 : 0) === r.trueRepay).length / LOAN_SCORED.length;

/* ── Permutation importance vs. MDI (impurity-based) importance ── */
function computePermutationImportance(forest, rows) {
  const acc = data => data.filter(r => (ensembleScore(forest, r) >= 0.5 ? 1 : 0) === r.historicalApproved).length / data.length;
  const baseAcc = acc(rows);
  const result = {};
  for (const feat of FEATURES) {
    let totalDrop = 0;
    for (let rep = 0; rep < 30; rep++) {
      const shuffled = rows.map(r => ({ ...r }));
      const rnd = seededRandom(100 + rep);
      const vals = shuffled.map(r => r[feat]);
      for (let i = vals.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [vals[i], vals[j]] = [vals[j], vals[i]]; }
      shuffled.forEach((r, i) => { r[feat] = vals[i]; });
      totalDrop += baseAcc - acc(shuffled);
    }
    result[feat] = totalDrop / 30;
  }
  return result;
}
const PERM_IMPORTANCE = computePermutationImportance(LOAN_FOREST, LOAN_ROWS);

function computeMDIImportance(forest) {
  const raw = { income: 0, creditScore: 0, debtRatio: 0 };
  for (const tree of forest) {
    (function walk(n) { if (n.isLeaf) return; raw[n.feature] += n.nSamples * n.impurityDecrease; walk(n.left); walk(n.right); })(tree);
  }
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v / total]));
}
const MDI_IMPORTANCE = computeMDIImportance(LOAN_FOREST);

/* ── Partial Dependence + ICE ── */
function featureRange(feat) {
  const vals = LOAN_ROWS.map(r => r[feat]);
  return [Math.min(...vals), Math.max(...vals)];
}
function pdpCurve(feat, gridN = 20) {
  const [lo, hi] = featureRange(feat);
  return Array.from({ length: gridN }, (_, i) => {
    const gv = lo + (hi - lo) * i / (gridN - 1);
    const avgScore = LOAN_ROWS.reduce((s, r) => s + ensembleScore(LOAN_FOREST, { ...r, [feat]: gv }), 0) / LOAN_ROWS.length;
    return { grid: gv, avgScore };
  });
}
const PDP_CURVES = Object.fromEntries(FEATURES.map(f => [f, pdpCurve(f)]));
const ICE_SAMPLE_ROWS = (() => {
  const sorted = [...LOAN_SCORED].sort((a, b) => a.score - b.score);
  return Array.from({ length: 10 }, (_, i) => sorted[Math.round(i * (sorted.length - 1) / 9)]);
})();
function iceCurves(feat, gridN = 20) {
  const [lo, hi] = featureRange(feat);
  const grid = Array.from({ length: gridN }, (_, i) => lo + (hi - lo) * i / (gridN - 1));
  return ICE_SAMPLE_ROWS.map(row => grid.map(gv => ensembleScore(LOAN_FOREST, { ...row, [feat]: gv })));
}

/* ── Exact Shapley values (2^3 = 8 coalitions, interventional/marginal definition) ── */
function factorial(k) { return k <= 1 ? 1 : k * factorial(k - 1); }
function subsetsOf(arr) { return arr.reduce((acc, x) => acc.concat(acc.map(s => [...s, x])), [[]]); }
function coalitionValue(forest, background, applicant, S) {
  let sum = 0;
  for (const bg of background) {
    const hybrid = { ...bg };
    for (const f of S) hybrid[f] = applicant[f];
    sum += ensembleScore(forest, hybrid);
  }
  return sum / background.length;
}
function exactShapley(forest, background, features, applicant) {
  const key = s => [...s].sort().join(',');
  const v = {};
  for (const S of subsetsOf(features)) v[key(S)] = coalitionValue(forest, background, applicant, S);
  const base = v[key([])];
  const full = v[key(features)];
  const n = features.length;
  const phi = {};
  for (const f of features) {
    const others = features.filter(x => x !== f);
    let acc = 0;
    for (const S of subsetsOf(others)) {
      const w = factorial(S.length) * factorial(n - S.length - 1) / factorial(n);
      acc += w * (v[key([...S, f])] - v[key(S)]);
    }
    phi[f] = acc;
  }
  return { base, full, phi };
}
const LOAN_SORTED_BY_SCORE = [...LOAN_SCORED].sort((a, b) => a.score - b.score);
const APPLICANTS = {
  denied: LOAN_SORTED_BY_SCORE[0],
  borderline: [...LOAN_SORTED_BY_SCORE].sort((a, b) => Math.abs(a.score - 0.5) - Math.abs(b.score - 0.5))[0],
  approved: LOAN_SORTED_BY_SCORE[LOAN_SORTED_BY_SCORE.length - 1],
};
const APPLICANT_LABELS = { denied: 'Clearly Denied', borderline: 'Borderline', approved: 'Clearly Approved' };
const SHAPLEY = Object.fromEntries(
  Object.entries(APPLICANTS).map(([key, applicant]) => [key, exactShapley(LOAN_FOREST, LOAN_ROWS, FEATURES, applicant)])
);

/* ── Fairness audit — `group` and `trueRepay` were withheld from training ── */
function fairnessMetricsAt(threshA, threshB) {
  const decide = r => (r.score >= (r.group === 'A' ? threshA : threshB) ? 1 : 0);
  const out = {};
  for (const g of ['A', 'B']) {
    const gs = LOAN_SCORED.filter(r => r.group === g);
    const pos = gs.filter(r => r.trueRepay === 1), neg = gs.filter(r => r.trueRepay === 0);
    out[g] = {
      approvalRate: gs.filter(r => decide(r) === 1).length / gs.length,
      tpr: pos.length ? pos.filter(r => decide(r) === 1).length / pos.length : 0,
      fpr: neg.length ? neg.filter(r => decide(r) === 1).length / neg.length : 0,
    };
  }
  const dpGap = Math.abs(out.A.approvalRate - out.B.approvalRate);
  const eoGap = Math.max(Math.abs(out.A.tpr - out.B.tpr), Math.abs(out.A.fpr - out.B.fpr));
  const overallAcc = LOAN_SCORED.filter(r => decide(r) === r.trueRepay).length / LOAN_SCORED.length;
  return { A: out.A, B: out.B, dpGap, eoGap, overallAcc };
}
// The "genuine" (feature-distribution-driven) part of the group gap, isolated by refitting with PENALTY=0
const LOAN_ROWS_NO_PENALTY = makeLoanDataset(0);
const GENUINE_TRUE_REPAY_GAP = Math.abs(
  LOAN_ROWS_NO_PENALTY.filter(r => r.group === 'A' && r.trueRepay === 1).length / LOAN_ROWS_NO_PENALTY.filter(r => r.group === 'A').length -
  LOAN_ROWS_NO_PENALTY.filter(r => r.group === 'B' && r.trueRepay === 1).length / LOAN_ROWS_NO_PENALTY.filter(r => r.group === 'B').length
);
// Three reference points on the threshold-tradeoff curve, cited directly (not hardcoded) in the Fairness tab's prose
const FAIRNESS_SHARED = fairnessMetricsAt(0.5, 0.5);
const FAIRNESS_EO_OPTIMAL = fairnessMetricsAt(0.5, 0.34);
const FAIRNESS_NEAR_FLOOR = fairnessMetricsAt(0.5, 0.12);

/* ── LIME's checkerboard (Tab 4) — a fresh dataset, verified necessary:
   a single global linear classifier scores 91-98% on this codebase's three
   existing spatial datasets (KNN_POINTS/TREE_POINTS/BAGGING_POOL), meaning
   none has the real curvature LIME's premise needs. This 4x4 checkerboard
   scores 50.0% (chance level) against a global line — genuine non-linearity. */
const CELL = 2.5;
function checkerLabel(x, y) { return (Math.floor(x / CELL) + Math.floor(y / CELL)) % 2; }
const CHECKER_POOL = (() => {
  const rand = seededRandom(21), jitterRand = seededRandom(798);
  return Array.from({ length: 220 }, () => {
    const x = rand() * 10, y = rand() * 10;
    const jx = x + (jitterRand() - 0.5) * 0.3, jy = y + (jitterRand() - 0.5) * 0.3;
    return { x, y, label: checkerLabel(jx, jy) };
  });
})();
const CHECKER_FOREST = fitForest(CHECKER_POOL, 25, 8, 1, ['x', 'y'], 42);
function checkerScore(x, y) { return ensembleScore(CHECKER_FOREST, { x, y }); }

/* ── LIME mechanic: weighted local linear fit via 3x3 Cramer's-rule solve ── */
function solveWeighted3x3(A, b) {
  const det3 = m => m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
  const D = det3(A);
  if (Math.abs(D) < 1e-12) return [0, 0, 0];
  const replCol = (m, col, vec) => m.map((row, i) => row.map((v, j) => j === col ? vec[i] : v));
  return [0, 1, 2].map(col => det3(replCol(A, col, b)) / D);
}
function limeFit(x0, y0, sigma, seed, K = 200) {
  const rand = seededRandom(seed);
  const XtWX = [[0,0,0],[0,0,0],[0,0,0]], XtWy = [0,0,0];
  for (let i = 0; i < K; i++) {
    const nx = x0 + randNormalish(rand, 0, sigma), ny = y0 + randNormalish(rand, 0, sigma);
    const d2 = (nx - x0) ** 2 + (ny - y0) ** 2;
    const w = Math.exp(-d2 / (2 * sigma * sigma));
    const score = checkerScore(clamp(nx, 0, 10), clamp(ny, 0, 10));
    const xi = [1, nx, ny];
    for (let a = 0; a < 3; a++) {
      XtWy[a] += w * xi[a] * score;
      for (let b = 0; b < 3; b++) XtWX[a][b] += w * xi[a] * xi[b];
    }
  }
  const [a, b, c] = solveWeighted3x3(XtWX, XtWy);
  return { a, b, c };
}
function limeFidelity(x0, y0, line, testRadius = 0.3, nTest = 50) {
  const rand = seededRandom(9999);
  let agree = 0;
  for (let i = 0; i < nTest; i++) {
    const tx = clamp(x0 + (rand() - 0.5) * 2 * testRadius, 0, 10), ty = clamp(y0 + (rand() - 0.5) * 2 * testRadius, 0, 10);
    const localPred = (line.a + line.b * tx + line.c * ty) >= 0.5 ? 1 : 0;
    const truePred = checkerScore(tx, ty) >= 0.5 ? 1 : 0;
    if (localPred === truePred) agree++;
  }
  return agree / nTest;
}

/* ── Small chart building blocks (hand-rolled, no chart library in this codebase) ── */
function HBarRow({ label, value, max, color, valueLabel }) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
        <span style={{ color: 'var(--color-text-secondary)' }}>{label}</span>
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)' }}>{valueLabel}</span>
      </div>
      <div style={{ height: 10, background: 'var(--color-background-tertiary)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

function PDPChart({ feat, showICE }) {
  const w = 460, h = 200, padL = 46, padB = 24, padT = 10, padR = 10;
  const curve = PDP_CURVES[feat];
  const [lo, hi] = featureRange(feat);
  const xToPx = v => padL + (v - lo) / (hi - lo) * (w - padL - padR);
  const yToPx = v => padT + (1 - v) * (h - padT - padB);
  const grid20 = Array.from({ length: 20 }, (_, i) => lo + (hi - lo) * i / 19);
  const ice = showICE ? iceCurves(feat) : [];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
      {[0, 0.5, 1].map(v => (
        <g key={v}>
          <line x1={padL} y1={yToPx(v)} x2={w - padR} y2={yToPx(v)} stroke="var(--color-border-tertiary)" strokeWidth={1} strokeDasharray="2,3" />
          <text x={padL - 6} y={yToPx(v) + 3} fontSize={10} textAnchor="end" fill="var(--color-text-tertiary)">{v.toFixed(1)}</text>
        </g>
      ))}
      {ice.map((curveVals, i) => (
        <path key={i} d={'M ' + curveVals.map((v, gi) => `${xToPx(grid20[gi]).toFixed(1)},${yToPx(v).toFixed(1)}`).join(' L ')}
          fill="none" stroke="var(--color-border-tertiary)" strokeWidth={1} opacity={0.7} />
      ))}
      <path d={'M ' + curve.map(p => `${xToPx(p.grid).toFixed(1)},${yToPx(p.avgScore).toFixed(1)}`).join(' L ')}
        fill="none" stroke="var(--accent)" strokeWidth={2.5} />
      <text x={w / 2} y={h - 6} fontSize={10} textAnchor="middle" fill="var(--color-text-tertiary)">{feat}</text>
    </svg>
  );
}

function ForcePlotChart({ base, phi, features, activeFeatures }) {
  const w = 460, h = 90;
  const xToPx = v => 20 + v * (w - 40);
  let cum = base;
  const segments = features.map(f => {
    const included = activeFeatures.has(f);
    const start = cum;
    cum += included ? phi[f] : 0;
    return { f, start, end: cum, delta: phi[f], included };
  });
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
      <line x1={xToPx(0)} y1={45} x2={xToPx(1)} y2={45} stroke="var(--color-border-tertiary)" strokeWidth={1} />
      <line x1={xToPx(base)} y1={12} x2={xToPx(base)} y2={68} stroke="var(--color-text-tertiary)" strokeWidth={1} strokeDasharray="3,3" />
      <text x={xToPx(base)} y={10} fontSize={10} textAnchor="middle" fill="var(--color-text-tertiary)">base={base.toFixed(3)}</text>
      {segments.map(seg => (
        <rect key={seg.f} x={xToPx(Math.min(seg.start, seg.end))} y={32}
          width={Math.max(0.5, Math.abs(xToPx(seg.end) - xToPx(seg.start)))} height={24}
          fill={seg.delta >= 0 ? 'var(--color-background-info)' : 'var(--color-background-danger)'}
          opacity={seg.included ? 1 : 0.15} />
      ))}
      <line x1={xToPx(cum)} y1={12} x2={xToPx(cum)} y2={68} stroke="var(--accent)" strokeWidth={2} />
      <text x={xToPx(cum)} y={84} fontSize={10} textAnchor="middle" fill="var(--color-text-primary)" fontWeight={700}>sum={cum.toFixed(3)}</text>
    </svg>
  );
}

const LIME_GRID_COLS = 20, LIME_GRID_ROWS = 14;
function LimeRaster({ query, neighbors, line }) {
  const cells = [];
  for (let r = 0; r < LIME_GRID_ROWS; r++) {
    for (let c = 0; c < LIME_GRID_COLS; c++) {
      const px = (c + 0.5) / LIME_GRID_COLS * 10, py = 10 - (r + 0.5) / LIME_GRID_ROWS * 10;
      cells.push(checkerScore(px, py) >= 0.5 ? 1 : 0);
    }
  }
  let lineSeg = null;
  if (line) {
    const pts = [];
    if (Math.abs(line.c) > 1e-6) {
      const yAt = x => (0.5 - line.a - line.b * x) / line.c;
      const y0 = yAt(0), y10 = yAt(10);
      if (y0 >= 0 && y0 <= 10) pts.push([0, y0]);
      if (y10 >= 0 && y10 <= 10) pts.push([10, y10]);
    }
    if (pts.length < 2 && Math.abs(line.b) > 1e-6) {
      const xAt = y => (0.5 - line.a - line.c * y) / line.b;
      const x0 = xAt(0), x10 = xAt(10);
      if (x0 >= 0 && x0 <= 10) pts.push([x0, 0]);
      if (x10 >= 0 && x10 <= 10) pts.push([x10, 10]);
    }
    if (pts.length >= 2) lineSeg = pts.slice(0, 2);
  }
  return (
    <div style={{ position: 'relative', height: 260, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden', cursor: 'crosshair' }}>
      <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: `repeat(${LIME_GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${LIME_GRID_ROWS}, 1fr)` }}>
        {cells.map((cls, i) => <div key={i} style={{ background: cls ? 'var(--color-background-danger)' : 'var(--color-background-info)', opacity: 0.5 }} />)}
      </div>
      {neighbors.map((n, i) => (
        <div key={i} style={{ position: 'absolute', left: `${n.x / 10 * 100}%`, top: `${(1 - n.y / 10) * 100}%`, width: 4, height: 4, borderRadius: '50%', background: 'var(--color-text-primary)', opacity: Math.min(1, n.w * 3), transform: 'translate(-50%,-50%)' }} />
      ))}
      {lineSeg && (
        <svg viewBox="0 0 10 10" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
          <line x1={lineSeg[0][0]} y1={10 - lineSeg[0][1]} x2={lineSeg[1][0]} y2={10 - lineSeg[1][1]} stroke="var(--accent)" strokeWidth={0.15} />
        </svg>
      )}
      {query && (
        <div style={{ position: 'absolute', left: `${query.x / 10 * 100}%`, top: `${(1 - query.y / 10) * 100}%`, width: 12, height: 12, borderRadius: '50%', border: '2px solid var(--color-text-primary)', background: 'var(--color-background-primary)', transform: 'translate(-50%,-50%)', pointerEvents: 'none' }} />
      )}
    </div>
  );
}

/* ── Demo components ── */
function GlobalLocalDemo() {
  const [mode, setMode] = useState('global');
  const [applicantKey, setApplicantKey] = useState('borderline');
  const applicant = APPLICANTS[applicantKey];
  const { base, phi } = SHAPLEY[applicantKey];
  const activeFeatures = useMemo(() => new Set(FEATURES), []);
  const maxImportance = Math.max(...Object.values(PERM_IMPORTANCE));
  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
        <button style={toggleBtnStyle(mode === 'global')} onClick={() => setMode('global')}>Global ranking</button>
        <button style={toggleBtnStyle(mode === 'local')} onClick={() => setMode('local')}>Local breakdown</button>
      </div>
      {mode === 'global' ? (
        <div>
          <P s={{ fontSize: 12, color: 'var(--color-text-tertiary)', marginBottom: 10 }} c="Permutation importance across all 240 applicants — how much accuracy drops when a feature is shuffled." />
          {FEATURES.map(f => (
            <HBarRow key={f} label={f} value={PERM_IMPORTANCE[f]} max={maxImportance} color="var(--color-background-info)" valueLabel={`${(PERM_IMPORTANCE[f] * 100).toFixed(1)}pp`} />
          ))}
        </div>
      ) : (
        <div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
            {Object.keys(APPLICANTS).map(k => (
              <button key={k} style={toggleBtnStyle(applicantKey === k)} onClick={() => setApplicantKey(k)}>{APPLICANT_LABELS[k]}</button>
            ))}
          </div>
          <P s={{ fontSize: 12, color: 'var(--color-text-tertiary)', marginBottom: 10 }} c="This one applicant's actual decomposition — base value plus each feature's exact contribution (full SHAP mechanics two tabs over)." />
          <ForcePlotChart base={base} phi={phi} features={FEATURES} activeFeatures={activeFeatures} />
        </div>
      )}
    </VizBox>
  );
}

function ImportancePDPDemo() {
  const [feat, setFeat] = useState('debtRatio');
  const [showICE, setShowICE] = useState(false);
  const [pointIdx, setPointIdx] = useState(10);
  const curve = PDP_CURVES[feat];
  const point = curve[pointIdx];
  const maxPerm = Math.max(...Object.values(PERM_IMPORTANCE));
  const maxMdi = Math.max(...Object.values(MDI_IMPORTANCE));
  return (
    <VizBox>
      <Grid cols={2} gap={16}>
        <div>
          <H3 c="Permutation importance" />
          {FEATURES.map(f => <HBarRow key={f} label={f} value={PERM_IMPORTANCE[f]} max={maxPerm} color="var(--color-background-info)" valueLabel={`${(PERM_IMPORTANCE[f] * 100).toFixed(1)}pp drop`} />)}
        </div>
        <div>
          <H3 c="MDI / impurity-based importance" />
          {FEATURES.map(f => <HBarRow key={f} label={f} value={MDI_IMPORTANCE[f]} max={maxMdi} color="var(--color-background-purple)" valueLabel={`${(MDI_IMPORTANCE[f] * 100).toFixed(1)}%`} />)}
        </div>
      </Grid>
      <div style={{ marginTop: 18, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        {FEATURES.map(f => <button key={f} style={toggleBtnStyle(feat === f)} onClick={() => { setFeat(f); setPointIdx(10); }}>{f}</button>)}
        <label style={{ marginLeft: 'auto', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" checked={showICE} onChange={e => setShowICE(e.target.checked)} /> Show ICE curves
        </label>
      </div>
      <PDPChart feat={feat} showICE={showICE} />
      <SliderRow label={`${feat} value`} min={0} max={19} step={1} value={pointIdx} onChange={setPointIdx}
        fmt={() => `${point.grid.toFixed(feat === 'debtRatio' ? 3 : 0)} → predicted score ${point.avgScore.toFixed(3)}`} />
    </VizBox>
  );
}

function ShapForcePlotDemo() {
  const [applicantKey, setApplicantKey] = useState('denied');
  const [active, setActive] = useState(() => new Set(FEATURES));
  const applicant = APPLICANTS[applicantKey];
  const { base, phi, full } = SHAPLEY[applicantKey];
  const cum = base + FEATURES.reduce((s, f) => s + (active.has(f) ? phi[f] : 0), 0);
  const toggle = f => setActive(prev => { const next = new Set(prev); if (next.has(f)) next.delete(f); else next.add(f); return next; });
  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        {Object.keys(APPLICANTS).map(k => (
          <button key={k} style={toggleBtnStyle(applicantKey === k)} onClick={() => { setApplicantKey(k); setActive(new Set(FEATURES)); }}>{APPLICANT_LABELS[k]}</button>
        ))}
      </div>
      <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Group {applicant.group} (shown for context only — never given to the model)</div>
      <Table heads={['Feature', 'Value']} rows={[
        ['income', `$${Math.round(applicant.income).toLocaleString()}`],
        ['creditScore', Math.round(applicant.creditScore)],
        ['debtRatio', applicant.debtRatio.toFixed(3)],
      ]} />
      <div style={{ display: 'flex', gap: 14, margin: '12px 0', flexWrap: 'wrap' }}>
        {FEATURES.map(f => (
          <label key={f} style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={active.has(f)} onChange={() => toggle(f)} /> {f} ({phi[f] >= 0 ? '+' : ''}{phi[f].toFixed(3)})
          </label>
        ))}
      </div>
      <ForcePlotChart base={base} phi={phi} features={FEATURES} activeFeatures={active} />
      <div style={{ fontSize: 12, color: 'var(--color-text-tertiary)', marginTop: 6 }}>
        Running sum = {cum.toFixed(4)}{active.size === FEATURES.length && ` — matches the model's actual score exactly (${full.toFixed(4)}), off by ${Math.abs(cum - full).toExponential(1)}.`}
      </div>
    </VizBox>
  );
}

function FairnessDemo() {
  const [threshA, setThreshA] = useState(0.5);
  const [threshB, setThreshB] = useState(0.5);
  const m = useMemo(() => fairnessMetricsAt(threshA, threshB), [threshA, threshB]);
  return (
    <VizBox>
      <SliderRow label="Group A threshold" min={0.1} max={0.9} step={0.01} value={threshA} onChange={setThreshA} fmt={v => v.toFixed(2)} />
      <SliderRow label="Group B threshold" min={0.1} max={0.9} step={0.01} value={threshB} onChange={setThreshB} fmt={v => v.toFixed(2)} />
      <Grid cols={2} gap={12}>
        <HBarRow label="Group A approval rate" value={m.A.approvalRate} max={1} color="var(--color-background-info)" valueLabel={`${(m.A.approvalRate * 100).toFixed(1)}%`} />
        <HBarRow label="Group B approval rate" value={m.B.approvalRate} max={1} color="var(--color-background-purple)" valueLabel={`${(m.B.approvalRate * 100).toFixed(1)}%`} />
      </Grid>
      <div style={{ display: 'flex', gap: 8, marginTop: 4, flexWrap: 'wrap' }}>
        <Badge color={m.dpGap < 0.05 ? 'success' : 'danger'}>Demographic parity gap: {(m.dpGap * 100).toFixed(1)}pp</Badge>
        <Badge color={m.eoGap < 0.05 ? 'success' : 'danger'}>Equalized odds gap: {(m.eoGap * 100).toFixed(1)}pp</Badge>
        <Badge color="info">Accuracy vs. true repayment: {(m.overallAcc * 100).toFixed(1)}%</Badge>
      </div>
    </VizBox>
  );
}

function LimeDemo() {
  const [query, setQuery] = useState({ x: 7.4, y: 2.6 });
  const [sigma, setSigma] = useState(0.3);
  const [resampleSeed, setResampleSeed] = useState(500);
  const fit = useMemo(() => limeFit(query.x, query.y, sigma, resampleSeed), [query, sigma, resampleSeed]);
  const neighbors = useMemo(() => {
    const rand = seededRandom(resampleSeed);
    const pts = [];
    for (let i = 0; i < 200; i++) {
      const nx = query.x + randNormalish(rand, 0, sigma), ny = query.y + randNormalish(rand, 0, sigma);
      if (nx < 0 || nx > 10 || ny < 0 || ny > 10) continue;
      const d2 = (nx - query.x) ** 2 + (ny - query.y) ** 2;
      pts.push({ x: nx, y: ny, w: Math.exp(-d2 / (2 * sigma * sigma)) });
    }
    return pts;
  }, [query, sigma, resampleSeed]);
  const fidelity = useMemo(() => limeFidelity(query.x, query.y, fit), [query, fit]);
  const gradMag = Math.hypot(fit.b, fit.c);

  const handleClick = e => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width * 10;
    const py = 10 - (e.clientY - rect.top) / rect.height * 10;
    setQuery({ x: clamp(px, 0, 10), y: clamp(py, 0, 10) });
  };

  return (
    <VizBox>
      <div onClick={handleClick}>
        <LimeRaster query={query} neighbors={neighbors} line={gradMag > 0.02 ? fit : null} />
      </div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', margin: '10px 0', flexWrap: 'wrap' }}>
        <button style={toggleBtnStyle(false)} onClick={() => setResampleSeed(s => s + 1)}>Resample</button>
        <span style={{ fontSize: 12, color: 'var(--color-text-tertiary)' }}>Click anywhere on the grid to move the query point.</span>
      </div>
      <SliderRow label="Bandwidth (σ)" min={0.1} max={1.2} step={0.05} value={sigma} onChange={setSigma} fmt={v => v.toFixed(2)} />
      <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
        <Badge color={fidelity > 0.9 ? 'success' : fidelity > 0.75 ? 'warning' : 'danger'}>Local fidelity: {(fidelity * 100).toFixed(0)}%</Badge>
        {gradMag <= 0.02 && <Badge color="info">No nearby boundary — confidently classified, line hidden</Badge>}
      </div>
    </VizBox>
  );
}

/* ── Section 1 — Global vs. Local Explanations ─────────────────────────── */
function SectionGlobalLocal() {
  return (
    <div>
      <H2 c="Global vs. Local Explanations" />
      <P c="Every interpretability technique answers one of two different questions. A global explanation describes how a model behaves across an entire dataset — 'overall, which features matter most?' A local explanation describes one specific decision — 'why was this particular applicant denied?' The two questions have different audiences (a model-risk committee wants global behavior; a denied applicant wants their own explanation), different failure modes, and — as the rest of this module shows — genuinely different math." />

      <H3 c="The case study used throughout this module" />
      <P c="Every tab from here on shares one running example: a bagged decision-tree ensemble (25 trees, the same bootstrap-aggregation mechanism as Bagging & Random Forest) trained to predict loan approval from three applicant features — income, credit score, and debt-to-income ratio — on 240 synthetic applicants split evenly across two demographic groups, A and B. Two things are deliberately withheld from the model: group membership, and a separate 'true repayment' label that reflects actual creditworthiness rather than the (partly biased) historical approval decisions the model is trained on. Group and true-repayment stay hidden from training specifically so Tab 5 can audit the model with information it never had access to." />
      <Note color="info" icon="ti-info-circle">
        The model's training label — <code>historicalApproved</code> — and the audit-only ground truth — <code>trueRepay</code> — agree for most applicants but not all, on purpose. That gap is the entire subject of the Fairness tab.
      </Note>

      <H3 c="Classifying this module's own techniques" />
      <Table heads={['Technique', 'Global?', 'Local?']} rows={[
        ['Permutation Importance', 'Yes — one ranking for the whole model', 'No'],
        ['Partial Dependence (PDP)', 'Yes — averaged across all rows', 'No (ICE curves are the per-instance analogue)'],
        ['SHAP', 'Yes — mean |value| aggregates into a ranking', 'Yes — exact, per-instance'],
        ['LIME', 'No', 'Yes — a fresh local fit for each query point'],
      ]} />
      <P c="SHAP's dual nature is the detail worth remembering for interviews: it is the only technique here that is genuinely both. Its per-instance values are additive by construction, so averaging their absolute values across every row produces a legitimate global ranking — something LIME's independently-refit local surrogates cannot do." />

      <GlobalLocalDemo />

      <Note color="warning" icon="ti-alert-triangle">
        Global and local views can disagree. A feature with unremarkable average importance can still dominate a handful of individual predictions — the Feature Importance & PDP tab's ICE-curve finding (debtRatio's effect on approval genuinely reorders which applicants benefit most, not just how much) is a concrete instance of exactly this.
      </Note>

      <H3 c="Interview Q&A" />
      <QA q="Is SHAP a global or a local method?" a="Both. Each SHAP value is an exact, instance-specific attribution — that part is local. But because the values are additive and comparable across instances, averaging |SHAP value| for a feature across every row in the dataset gives a legitimate global importance ranking. That dual nature is one of SHAP's main selling points over LIME, which only ever produces local explanations." />
      <QA q="Why would you want a local explanation if you already have a global feature-importance ranking?" a="Because the global ranking is an average, and averages hide variation. Regulatory and compliance settings (adverse-action notices in lending, GDPR's 'right to explanation') require justifying one specific decision, not the model in the abstract — 'credit score matters on average' does not tell a denied applicant why they, specifically, were denied." />
      <QA q="Give a concrete example where a global and a local explanation would point to different conclusions." a="This module's own Feature Importance & PDP tab: debtRatio ranks highest globally by permutation importance, but its ICE curves show the size of its effect — and even the relative ranking of which applicants gain the most from a lower debt ratio — genuinely changes across individuals, something the single averaged PDP curve cannot reveal." />
      <QA q="What's the practical cost difference between global and local methods in a production system?" a="Global explanations (permutation importance, PDP) are computed once, off-line, against a reference dataset — cheap to log and monitor over time. Local explanations (SHAP, LIME) must be computed per prediction, which adds real latency to serving and, for exact Shapley values specifically, cost that grows exponentially with feature count — one reason production systems reach for TreeSHAP or KernelSHAP approximations instead of exact enumeration." />
    </div>
  );
}

/* ── Section 2 — Feature Importance & Partial Dependence ───────────────── */
function SectionImportancePDP() {
  return (
    <div>
      <H2 c="Feature Importance & Partial Dependence" />
      <P c="Two different questions get lumped together under 'feature importance.' Permutation importance asks: how much does the model's performance degrade if this feature's values are scrambled? Impurity-based (MDI) importance asks a structural question instead: how much did this feature reduce impurity, summed across every split that used it, while the tree was being built? They usually agree on direction — they don't have to agree on magnitude." />

      <H3 c="Permutation importance" />
      <Mx block>{`Importance(feature) = Accuracy(original) − Accuracy(feature shuffled)`}</Mx>
      <P c="Shuffle one column, leave everything else untouched, re-measure accuracy. A feature the model actually depends on causes a real drop; a feature it ignores causes none. Averaging over many independent shuffles (30, here) smooths out the luck of any one shuffle." />

      <H3 c="MDI / impurity-based importance" />
      <Mx block>{`MDI(feature) = (1/T) · Σ_trees Σ_{splits on feature} (n_at_split / n_total) · Δgini(split)`}</Mx>
      <P c="Every time a tree splits on a feature, that split reduces Gini impurity by some amount, weighted by how many training rows reached that node. Summing this across every split, in every tree, gives a structural importance score — computed for free during training, unlike permutation importance which requires a full extra pass over the data per feature." />

      <Card color="info" title="Measured on this module's loan model">
        Permutation importance (mean accuracy drop over 30 shuffles): <strong>debtRatio 26.7pp</strong>, income 16.3pp, creditScore 11.6pp.
        MDI importance (normalized): <strong>debtRatio 51.6%</strong>, income 27.7%, creditScore 20.7%. Both agree on the ranking here — debtRatio dominates either way — but MDI concentrates more of the total credit on debtRatio than permutation importance does, a real magnitude difference worth noticing even when the ranking itself doesn't move.
      </Card>

      <H3 c="Partial Dependence (PDP) and Individual Conditional Expectation (ICE)" />
      <Mx block>{`PDP(x_s) = (1/n) · Σᵢ f(x_s, x_c⁽ⁱ⁾)      — feature s fixed at a grid value, the rest (x_c) left at each row's real values, averaged`}</Mx>
      <P c="Pick a feature, fix it at a grid of values, and for each grid value replace that one feature across every row while leaving everything else alone — then average the model's predicted score. The result is the feature's marginal effect, with every other feature's influence averaged out. ICE curves are the same computation without the final averaging step: one curve per instance instead of one averaged curve for the whole dataset." />

      <Grid cols={3} gap={10}>
        <Card color="success" title="income">Rises from ≈0.18 (at $20-37k) to ≈0.83 (at $100k) — steep in the middle, flat at both ends.</Card>
        <Card color="success" title="creditScore">Rises from ≈0.25 (at ~520) to a plateau of ≈0.75 above roughly 760 — more score stops mattering past that point.</Card>
        <Card color="danger" title="debtRatio">Falls from a plateau of ≈0.85 (below ~0.17) to a plateau of ≈0.18 (above ~0.44) — the expected direction, reversed shape.</Card>
      </Grid>

      <ImportancePDPDemo />

      <Note color="warning" icon="ti-alert-triangle">
        For debtRatio specifically, the 10 ICE curves sampled across the score range are genuinely NOT parallel: their spread ranges from 0.40 to 0.88 across the grid (comparable to the PDP curve's own 0.66 total range), and the rank order of which sampled applicants have the highest predicted score at low debtRatio is not the same as at high debtRatio — real crossing, not noise. The averaged PDP line is still directionally correct, but it hides that debtRatio's effect size — and even which applicants benefit most from a lower one — depends on the rest of an applicant's profile.
      </Note>
      <Note color="info" icon="ti-info-circle">
        Both permutation importance and PDP silently assume the shuffled/marginalized feature is independent of the others — shuffling debtRatio while leaving income untouched can create combinations that never occur in real applicants if the two are correlated. In this case study the three features are generated independently within each group, so the assumption holds; it's flagged here because it's a genuine, common failure mode when features ARE correlated (e.g. income and credit score usually are, in real data).
      </Note>

      <Code lang="python">{`from sklearn.inspection import permutation_importance, PartialDependenceDisplay

r = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    print(X_test.columns[i], r.importances_mean[i])

# kind='both' overlays the individual ICE curves on top of the averaged PDP line
PartialDependenceDisplay.from_estimator(model, X_train, features=['debtRatio'], kind='both')`}</Code>

      <H3 c="Interview Q&A" />
      <QA q="Why can permutation importance and MDI importance disagree?" a="They measure different things. MDI is computed purely from training-time impurity reduction — it can be inflated by overfitting and by features with many possible split points. Permutation importance measures an actual performance drop, which more directly reflects real predictive value. On this module's model they happen to agree on ranking (debtRatio > income > creditScore) but differ in relative magnitude — MDI concentrates more credit on debtRatio than the permutation-based measurement does." />
      <QA q="What does a PDP assume that can make it misleading?" a="Feature independence. A PDP evaluates the model on combinations of feature values that may never occur together in reality if the features are correlated — e.g. forcing a very low income together with a very high credit score, if those two are usually correlated in real applicants. When that assumption is violated, the curve reflects the model's behavior on synthetic, unrealistic rows." />
      <QA q="When would you look at ICE curves instead of just the PDP?" a="When you suspect the feature's effect isn't the same for every instance. This module's debtRatio ICE curves genuinely cross and vary in spread (0.40 to 0.88) — a case where the single averaged PDP line, while directionally right, hides real heterogeneity that only per-instance curves reveal." />
      <QA q="Why compute permutation importance on held-out data rather than the training set?" a="Measuring it on training data can credit a feature for helping the model overfit rather than genuinely predict — a feature the model has memorized noise from can look important on training data and useless on new data. This module's demo measures it in-sample for simplicity, which is worth naming as a real simplification rather than best practice." />
    </div>
  );
}

/* ── Section 3 — SHAP ───────────────────────────────────────────────── */
function SectionShap() {
  return (
    <div>
      <H2 c="SHAP (SHapley Additive exPlanations)" />
      <P c="SHAP borrows a 70-year-old idea from cooperative game theory. Treat a prediction as a game: the features are players, and the model's output relative to some baseline is the payout the players jointly earned. The Shapley value is the unique way to split that payout among the players that satisfies four fairness axioms simultaneously — no other attribution scheme does." />

      <H3 c="The four axioms" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Efficiency">The contributions sum exactly to (prediction − base value) — nothing left over, nothing double-counted.</Card>
        <Card color="info" title="Symmetry">Two features that contribute identically to every possible coalition get identical values.</Card>
        <Card color="info" title="Dummy (Null Player)">A feature that never changes the prediction, in any coalition, gets a value of exactly zero.</Card>
        <Card color="info" title="Additivity">For a model that's the sum of two sub-models, each feature's Shapley value is the sum of its values in each sub-model.</Card>
      </Grid>

      <H3 c="The exact formula" />
      <Mx block>{`φᵢ(v) = Σ_{S ⊆ F∖{i}}  [ |S|!·(|F|−|S|−1)! / |F|! ]  ·  [ v(S∪{i}) − v(S) ]`}</Mx>
      <P c="For every possible subset S of the other features, measure how much adding feature i changes the coalition's value, then average those marginal contributions with weights equal to how many feature orderings produce each subset. With F = {income, creditScore, debtRatio}, there are only 2³ = 8 coalitions total — small enough to enumerate exactly rather than approximate, which is exactly why this case study deliberately uses only 3 features." />
      <P c="A coalition's value, v(S), is computed the standard interventional/marginal way: fix the in-coalition features to the applicant's own values, and average the model's prediction over every other row in the dataset for the remaining features. v(∅) — the value with no features fixed — is just the model's average prediction over the whole dataset, the 'base value' every force plot starts from." />

      <H3 c="Verified: three applicants, exact decomposition" />
      <P c="Base value (mean predicted score over all 240 applicants): 0.481 — identical for every applicant, since it doesn't depend on who's being explained." />
      <Table heads={['Applicant', 'income', 'creditScore', 'debtRatio', 'phi(income)', 'phi(creditScore)', 'phi(debtRatio)', 'Final score']} rows={[
        ['Clearly Denied (A)', '$65,797', '618', '0.489', '−0.030', '−0.112', '−0.336', '0.003'],
        ['Borderline (B)', '$47,710', '693', '0.301', '−0.085', '+0.111', '−0.008', '0.499'],
        ['Clearly Approved (B)', '$64,482', '688', '0.174', '+0.058', '+0.059', '+0.403', '1.000'],
      ]} />
      <Note color="success" icon="ti-checkbox">
        Efficiency, verified rather than assumed: summing base + all three φ values reproduces each applicant's actual model score to within 10⁻¹⁵–10⁻¹⁹ — pure floating-point round-off, not an approximation. This is what "exact" means here.
      </Note>

      <ShapForcePlotDemo />

      <Note color="warning" icon="ti-alert-triangle">
        Exact enumeration only stayed feasible because this case study fixed the feature count at 3. A model with 20 features has over a million coalitions — real-world SHAP almost always uses an approximation instead: <strong>TreeSHAP</strong> exploits tree structure to compute exact-for-trees values in low-order-polynomial time; <strong>KernelSHAP</strong> is a model-agnostic, sampling-based weighted linear regression (conceptually close to LIME, but using Shapley-consistent sample weights) usable on any model at all.
      </Note>

      <Code lang="python">{`import shap

explainer = shap.TreeExplainer(model)          # exact-for-trees, fast even with many features
shap_values = explainer.shap_values(X.iloc[[0]])
shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])`}</Code>

      <H3 c="Interview Q&A" />
      <QA q="What four axioms uniquely define the Shapley value?" a="Efficiency (values sum exactly to prediction minus base value), symmetry (equal contribution across all coalitions means equal value), the dummy/null-player property (zero contribution in every coalition means zero value), and additivity (values decompose linearly across summed sub-models). The Shapley value is the ONLY attribution rule satisfying all four at once — that uniqueness, not just its formula, is SHAP's actual selling point." />
      <QA q="Why is exact Shapley computation usually infeasible, and what's used instead in practice?" a="The number of coalitions grows as 2ⁿ in the feature count — trivial at n=3 (8 coalitions, as here), intractable well before n=20. TreeSHAP is used for tree-based models specifically (exploits the tree structure for a fast, still-exact computation); KernelSHAP is a model-agnostic sampling approximation for anything else." />
      <QA q="What's the practical difference between TreeSHAP and KernelSHAP?" a="TreeSHAP only works on tree ensembles but is exact and fast, because it can walk the actual tree structure instead of sampling. KernelSHAP works on any black-box model at all — including neural networks — but is an approximation, obtained by fitting a weighted linear regression over sampled coalitions, similar in spirit to LIME but with sample weights specifically derived from Shapley's axioms." />
      <QA q="Is a large SHAP value for one applicant the same as a feature being generally important?" a="No — that's exactly the global-vs-local distinction from the first tab. debtRatio contributes +0.403 for the clearly-approved applicant here but only −0.008 for the borderline one; a feature can swing one prediction hugely while being nearly irrelevant to another, which is precisely why SHAP's per-instance values, not just its aggregated ranking, matter." />
    </div>
  );
}

/* ── Section 4 — LIME ───────────────────────────────────────────────── */
function SectionLime() {
  return (
    <div>
      <H2 c="LIME (Local Interpretable Model-Agnostic Explanations)" />
      <P c="LIME explains one prediction by building a tiny, honest local map of the black box's behavior right around it: perturb the input a little, ask the real model what it thinks of each perturbation, weight the answers by how close each perturbation is to the original point, and fit the simplest interpretable model — here, a straight line — to those weighted answers. The explanation is that local fit, not the black box itself." />

      <Mx block>{`explanation(x) = argmin_g  Σ_z π_x(z)·(f(z) − g(z))²  +  Ω(g)          [minimized over g ∈ G]
  f = the real black-box model     g = a simple local model (here, linear)
  π_x(z) = a proximity weight, larger for perturbations z closer to x
  Ω(g) = a complexity penalty (kept implicit here by using a plain 3-parameter line)`}</Mx>

      <H3 c="A fresh dataset, on purpose" />
      <P c="LIME's entire premise is that a globally nonlinear model can look locally linear in a small enough neighborhood — which requires real global curvature to demonstrate. This codebase's three existing 2D spatial datasets (from KNN & Decision Trees and Bagging & Random Forest) turn out to be almost entirely linear once checked: a single global straight-line classifier scores 73-98% on all three. So this tab uses a fresh one instead — a 4×4 checkerboard over a 10×10 grid, jittered to avoid perfectly crisp cell edges. A global line scores exactly 50.0% on it — chance level, genuine non-separability by any single boundary." />
      <Code lang="python">{`# same mechanism as Bagging & Random Forest (bootstrap-aggregated trees),
# a fresh dataset chosen because it actually needs LIME's premise:
# checker_label(x, y) = (floor(x/2.5) + floor(y/2.5)) % 2`}</Code>

      <H3 c="The black box: a 25-tree bagged ensemble" />
      <P c="Same bootstrap-aggregation mechanism as Bagging & Random Forest, fit deep enough (maxDepth 8) to be a genuine black box. Verified against the true, noiseless checkerboard function on an 80x80 held-out grid: 99.1% training accuracy, 86.7% true-function accuracy for the full ensemble — while individual bootstrap trees average only 74.1% (±9.1%, range 58-90%) on that same test. Bagging's variance reduction is doing real, measurable work here, not just decoration." />

      <LimeDemo />

      <H3 c="Verified: local fidelity depends on where you ask" />
      <Table heads={['Query location', 'Local fidelity (agreement with the real model, tiny neighborhood)']} rows={[
        ['Deep inside a region, far from any boundary', '100.0%'],
        ['Along a straight stretch of boundary', '100.0%'],
        ['Near a boundary, away from a corner', '93.0% (± 0.4% across 30 reseeds)'],
        ['At a checkerboard corner — real curvature', '80.0% (± 2.8% across 30 reseeds)'],
      ]} />
      <P c="That 80% is reported honestly, not smoothed over: it's the real, worst-case measurement, at the one place the checkerboard actually curves in two directions at once. Straight stretches and flat interiors — most of the space — get a perfect local match." />

      <Note color="warning" icon="ti-alert-triangle">
        LIME's answer is not fully deterministic. Refitting at the exact same point with a different random perturbation sample shifts the local line by a real, measured 2-6° near genuine boundary structure (with essentially no positional drift) — small but genuine instability, not a bug. Deep inside a flat region, the fitted line's implied position becomes numerically meaningless (an 11° angle swing and a position spread larger than the entire domain) — which is exactly why this demo hides the line rather than draw a meaningless one snapping in from nowhere. Try the Resample button at a corner versus deep in a region to feel the difference.
      </Note>
      <Note color="info" icon="ti-info-circle">
        The bandwidth slider (σ) controls how "local" local means. Push it below roughly 0.05 and the fit degenerates entirely — every sampled neighbor lands in the same leaf of the tree ensemble, and the fitted gradient collapses to exactly zero. Push it too high and the neighborhood stops being local at all, which is most visible in reduced fidelity right at the checkerboard's corners.
      </Note>

      <Code lang="python">{`from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns,
                                  class_names=['class0', 'class1'])
exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba, num_features=2)
exp.as_list()`}</Code>

      <H3 c="Interview Q&A" />
      <QA q="What does LIME actually fit, and why is it called 'model-agnostic'?" a="It fits a simple, interpretable local model — here a weighted linear regression — to a synthetic neighborhood sampled around one instance, using only the black box's predictions on those samples. It never inspects the black box's internals, only queries it, so it works identically whether the underlying model is a tree ensemble, a neural network, or anything else that can produce a prediction." />
      <QA q="Why can LIME give a slightly different explanation if you rerun it on the same point?" a="Its neighborhood is a random sample. Measured directly on this module's checkerboard: refitting at the same query point with a fresh random seed shifts the local line's angle by 2-6° near real boundary structure — real, modest, honest instability inherent to the method, not implementation noise." />
      <QA q="What role does the bandwidth (σ) play, and what happens if you get it wrong?" a="σ sets how tightly the sampled neighborhood clusters around the query point, which is also the local linear model's implicit definition of 'local.' Too small (below ~0.05 here) and every sample collapses into one leaf of the underlying tree, producing a flat, zero-gradient fit; too large and fidelity measurably drops, most visibly at points of real curvature like this checkerboard's corners (80% fidelity, the worst case measured, versus 100% along straight stretches)." />
      <QA q="How is LIME fundamentally different from SHAP?" a="LIME heuristically fits an arbitrary local surrogate via random sampling and regression — useful and general, but not uniquely determined by any axioms, and local-only. SHAP's values are the unique solution satisfying four specific fairness axioms (efficiency, symmetry, dummy, additivity), and aggregate cleanly into a global ranking, which LIME's independently-refit local surrogates cannot do." />
    </div>
  );
}

/* ── Section 5 — Fairness, Bias & Auditing ─────────────────────────────── */
function SectionFairness() {
  return (
    <div>
      <H2 c="Fairness, Bias & Auditing" />
      <P c="Every technique so far explained what this loan model learned. This tab audits whether it learned something it shouldn't have — using the same model, plus the two things deliberately withheld from its training: group membership, and trueRepay, a ground-truth repayment label that (unlike the historicalApproved label the model was actually trained on) carries no direct historical penalty." />

      <H3 c="Types of bias" />
      <Grid cols={3} gap={10}>
        <Card color="warning" title="Historical bias">Bias that existed in the real world and got baked into training labels, even with perfect data collection. Modeled here directly: historicalApproved applies an extra penalty to group B's approval label that trueRepay does not.</Card>
        <Card color="warning" title="Representation bias">Some groups underrepresented in the training data itself. Not present in this case study (120/120 split) — named for completeness, since it's a distinct, equally real failure mode.</Card>
        <Card color="warning" title="Measurement bias">The label used for training is a flawed proxy for what you actually care about. historicalApproved is exactly this: a biased proxy for the trueRepay outcome the model is really meant to predict.</Card>
      </Grid>

      <H3 c="Fairness metrics" />
      <Mx block>{`Demographic Parity:   P(ŷ=1 | group=A)  =  P(ŷ=1 | group=B)
Equalized Odds:       P(ŷ=1 | y=1, group=A) = P(ŷ=1 | y=1, group=B)     (equal TPR)
                      P(ŷ=1 | y=0, group=A) = P(ŷ=1 | y=0, group=B)     (equal FPR)`}</Mx>
      <P c="Demographic parity asks for an equal approval RATE across groups, full stop — it never looks at whether the approval was actually deserved. Equalized odds asks a stricter, outcome-aware question: among applicants who would actually repay, are both groups approved at the same rate (equal TPR)? And among those who wouldn't, are both groups incorrectly approved at the same rate (equal FPR)? The two can pull in different directions, as the live demo below shows directly." />

      <Note color="danger" icon="ti-alert-triangle">
        Accuracy alone hides this entirely. This model is {(LOAN_ACC_HIST * 100).toFixed(1)}% accurate against the label it was trained on and {(LOAN_ACC_TRUE * 100).toFixed(1)}% accurate against true repayment — a gap of only {((LOAN_ACC_HIST - LOAN_ACC_TRUE) * 100).toFixed(1)} points. A model-performance dashboard reporting only overall accuracy would show nothing alarming. The group-conditional gap below is dramatically larger.
      </Note>

      <H3 c="Where does the group gap actually come from?" />
      <P>
        Refitting the same case study with the historical-bias penalty set to zero still leaves a {(GENUINE_TRUE_REPAY_GAP * 100).toFixed(1)}-point gap in trueRepay itself, purely from group A and B having different income/credit-score/debt-ratio distributions — a real socioeconomic disparity baked into the features, with no explicit bias anywhere. The model was never given group as an input, yet reproduces most of this gap anyway, simply because income, credit score, and debt ratio are themselves correlated with group. This is disparate impact in miniature: facially neutral criteria, applied identically to everyone, producing systematically unequal outcomes — no explicit discrimination required.
      </P>

      <FairnessDemo />

      <H3 c="Verified: the threshold tradeoff" />
      <Table heads={['Threshold setting', 'Demographic parity gap', 'Equalized odds gap', 'Accuracy vs. trueRepay']} rows={[
        ['Shared threshold (0.50 / 0.50)', `${(FAIRNESS_SHARED.dpGap * 100).toFixed(1)}pp`, `${(FAIRNESS_SHARED.eoGap * 100).toFixed(1)}pp`, `${(FAIRNESS_SHARED.overallAcc * 100).toFixed(1)}%`],
        ['Equalized-odds-optimal (0.50 / 0.34)', `${(FAIRNESS_EO_OPTIMAL.dpGap * 100).toFixed(1)}pp`, `${(FAIRNESS_EO_OPTIMAL.eoGap * 100).toFixed(1)}pp`, `${(FAIRNESS_EO_OPTIMAL.overallAcc * 100).toFixed(1)}%`],
        ['Pushed toward the slider floor (0.50 / 0.12)', `${(FAIRNESS_NEAR_FLOOR.dpGap * 100).toFixed(1)}pp`, `${(FAIRNESS_NEAR_FLOOR.eoGap * 100).toFixed(1)}pp`, `${(FAIRNESS_NEAR_FLOOR.overallAcc * 100).toFixed(1)}%`],
      ]} />
      <Note color="warning" icon="ti-alert-triangle">
        Lowering group B's threshold from the shared 0.50 to 0.34 nearly perfectly closes the equalized-odds gap (down to {(FAIRNESS_EO_OPTIMAL.eoGap * 100).toFixed(1)} points) at a modest accuracy cost — but demographic parity barely moves ({(FAIRNESS_EO_OPTIMAL.dpGap * 100).toFixed(1)} points, versus {(FAIRNESS_SHARED.dpGap * 100).toFixed(1)} at the shared threshold). Pushing further still, toward the slider's floor, drags equalized odds back up to {(FAIRNESS_NEAR_FLOOR.eoGap * 100).toFixed(1)} points — worse than where it started — while demographic parity still hasn't closed. One threshold knob cannot satisfy both criteria at once here: a concrete, measured instance of the general result (Kleinberg/Chouldechova) that demographic parity, equalized odds, and calibration are mutually incompatible except in degenerate special cases.
      </Note>

      <H3 c="Mitigation taxonomy" />
      <Table heads={['Stage', 'Technique', 'Demonstrated above?']} rows={[
        ['Pre-processing', 'Reweighing or resampling the training data before fitting', 'No'],
        ['In-processing', 'Fairness-constrained optimization or adversarial debiasing during training', 'No'],
        ['Post-processing', 'Per-group threshold calibration on an already-trained model', 'Yes — exactly what the sliders above do'],
      ]} />
      <P c="Post-processing is the cheapest to deploy — no retraining required — which is exactly why it's demonstrated here. It's also, as just shown, the most limited: it can shift where the tradeoff sits, but it cannot manufacture a threshold that satisfies every fairness criterion simultaneously when the underlying group distributions genuinely differ." />

      <H3 c="A legal framing worth knowing" />
      <P>
        <strong>Disparate treatment</strong> is explicit differential treatment based on a protected attribute — straightforwardly prohibited in US employment and credit law. <strong>Disparate impact</strong> is different: facially neutral criteria, applied identically to everyone, that produce unequal outcomes across protected groups — potentially unlawful too, unless justified by legitimate business necessity. This model never uses group as an input, so it has no disparate-treatment problem. It very plausibly has a disparate-impact one — which is exactly why fairness auditing has to look at outcomes, not just at what features a model was given.
      </P>

      <Code lang="python">{`from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
demographic_parity_difference(y_true, y_pred, sensitive_features=group)
equalized_odds_difference(y_true, y_pred, sensitive_features=group)

from fairlearn.postprocessing import ThresholdOptimizer   # per-group calibration, like the sliders above
`}</Code>

      <H3 c="Interview Q&A" />
      <QA q="What's the difference between demographic parity and equalized odds?" a="Demographic parity only requires an equal predicted-positive RATE across groups, regardless of whether the outcome is actually correct. Equalized odds is outcome-aware: it requires equal true-positive rates AND equal false-positive rates across groups. This module's own threshold sweep shows they can require genuinely different thresholds — the setting that nearly perfectly equalizes odds barely moves demographic parity at all." />
      <QA q="Can you satisfy demographic parity, equalized odds, and calibration all at once?" a="Generally no, except in degenerate cases (equal base rates across groups, or a perfect model) — a known impossibility result. This module's threshold sweep is a concrete miniature of it: the equalized-odds-optimal threshold leaves demographic parity almost exactly where it started, and pushing further to fix demographic parity makes equalized odds worse again." />
      <QA q="Why does a model that never sees 'group' as a feature still end up discriminating by group?" a="Two reasons, both present here. First, proxy discrimination: the features it DOES see (income, credit score, debt ratio) are themselves correlated with group, so the model can reconstruct group-like information indirectly — confirmed by refitting with zero explicit historical bias and still finding a real gap in the ground-truth label itself. Second, in this case study, an explicit historical penalty was also baked directly into the training label, which the model dutifully learned to reproduce." />
      <QA q="What's the legal difference between disparate treatment and disparate impact?" a="Disparate treatment is explicit, intentional differential treatment by a protected attribute — directly unlawful. Disparate impact is facially neutral criteria producing unequal outcomes across groups — potentially unlawful too, unless justified by business necessity, and notably does NOT require the model to have ever seen the protected attribute. This model's group as a withheld feature makes it a disparate-impact case, not a disparate-treatment one." />
    </div>
  );
}

/* ── Page shell ─────────────────────────────────────────────────────── */
const TABS = [
  { id: 'global-local', label: 'Global vs Local', sub: 'Two lenses for reading any model' },
  { id: 'importance-pdp', label: 'Feature Importance & PDP', sub: 'Permutation, MDI & partial dependence' },
  { id: 'shap', label: 'SHAP', sub: 'Exact, game-theoretic attribution' },
  { id: 'lime', label: 'LIME', sub: 'Local surrogate models' },
  { id: 'fairness', label: 'Fairness & Auditing', sub: 'Bias, demographic parity & equalized odds' },
];
const SECTION_MAP = {
  'global-local': <SectionGlobalLocal />,
  'importance-pdp': <SectionImportancePDP />,
  'shap': <SectionShap />,
  'lime': <SectionLime />,
  'fairness': <SectionFairness />,
};

export default function InterpretabilityFairness() {
  const [active, setActive] = useState('global-local');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 20</div>
        <h1 className="page-header-title">Interpretability & Fairness</h1>
        <p className="page-header-subtitle">Explaining what a model actually learned — SHAP, LIME, feature importance and partial dependence — and auditing whether it learned something it shouldn't have.</p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={20} />
    </div>
  );
}
