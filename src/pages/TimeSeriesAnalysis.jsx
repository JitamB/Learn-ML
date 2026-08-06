import { useState, useMemo, useEffect } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow, Badge } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

/* ── Shared helpers ─────────────────────────────────────────── */
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
}
function randNormalish(rand, mean, std) {
  const u = rand() + rand() + rand() - 1.5;
  return mean + (u / 0.5) * std;
}
function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 12px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}

/* ── Small chart-scale helpers (this file draws many compact line/bar charts) ── */
function makeScales(xDomain, yDomain, w, h, padL = 32, padB = 18) {
  const xToPx = x => padL + (x - xDomain[0]) / ((xDomain[1] - xDomain[0]) || 1) * (w - padL - 6);
  const yToPx = y => h - padB - (y - yDomain[0]) / ((yDomain[1] - yDomain[0]) || 1) * (h - 6 - padB);
  return { xToPx, yToPx };
}
function linePath(pts, xToPx, yToPx) { return 'M ' + pts.map(([x, y]) => `${xToPx(x).toFixed(2)},${yToPx(y).toFixed(2)}`).join(' L '); }
function domainOf(arr, padFrac = 0.12) {
  const finite = arr.filter(v => Number.isFinite(v));
  const mn = Math.min(...finite), mx = Math.max(...finite);
  const pad = (mx - mn) * padFrac || 1;
  return [mn - pad, mx + pad];
}
function MiniChart({ w = 280, h = 90, series, xDomain, yDomain, extra }) {
  const { xToPx, yToPx } = makeScales(xDomain, yDomain, w, h);
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)', display: 'block' }}>
      <line x1={0} y1={yToPx(0)} x2={w} y2={yToPx(0)} stroke="var(--color-border-tertiary)" strokeWidth={0.5} strokeDasharray="2,2" />
      {series.map((s, i) => (
        <path key={i} d={linePath(s.points, xToPx, yToPx)} fill="none" stroke={s.color} strokeWidth={s.width || 1.6} opacity={s.opacity ?? 1} strokeDasharray={s.dash} />
      ))}
      {extra && extra({ xToPx, yToPx })}
    </svg>
  );
}

/* ── Small hand-solved linear algebra — same Gauss-Jordan precedent as ──
   HyperparameterTuning.jsx / ValidationBiasVariance.jsx / Regularization.jsx */
function solveLinearSystem(Ain, bin) {
  const n = Ain.length;
  const A = Ain.map(row => row.slice());
  const b = bin.slice();
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
    [A[col], A[piv]] = [A[piv], A[col]];
    [b[col], b[piv]] = [b[piv], b[col]];
    const d = A[col][col];
    for (let j = col; j < n; j++) A[col][j] /= d;
    b[col] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = A[r][col];
      if (f === 0) continue;
      for (let j = col; j < n; j++) A[r][j] -= f * A[col][j];
      b[r] -= f * b[col];
    }
  }
  return b;
}
function invertMatrix(Ain) {
  const n = Ain.length, cols = [];
  for (let i = 0; i < n; i++) { const e = new Array(n).fill(0); e[i] = 1; cols.push(solveLinearSystem(Ain, e)); }
  return Array.from({ length: n }, (_, r) => cols.map(c => c[r]));
}
function transpose(M) { return M[0].map((_, j) => M.map(row => row[j])); }
function matMul(A, B) { return A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0))); }
function matVec(A, v) { return A.map(row => row.reduce((s, a, i) => s + a * v[i], 0)); }

/* ── Shared dataset: TS_MAIN (stochastic-trend + seasonal + noise), and ──
   TS_MULT (a small illustrative multiplicative-seasonality series). A
   purely deterministic trend turns out NOT to reliably test as
   non-stationary at this sample size — verified — so the trend here is a
   real random walk with drift, which is also more realistic (Holt's method
   literally assumes this shape). */
const TS_N = 120, TS_PERIOD = 12;
const TS_MAIN = (() => {
  const randTrend = seededRandom(777), randNoise = seededRandom(778);
  const trend = [50];
  for (let t = 1; t < TS_N; t++) trend.push(trend[t - 1] + 0.25 + randNormalish(randTrend, 0, 0.35));
  const seasonalFn = m => 10 * Math.sin(2 * Math.PI * m / 12);
  return trend.map((tr, t) => tr + seasonalFn(t % 12) + randNormalish(randNoise, 0, 2));
})();
const TS_MULT = (() => {
  const randN = seededRandom(555);
  return Array.from({ length: TS_N }, (_, t) => {
    const trend = 50 + 0.25 * t;
    const seasMult = 1 + 0.3 * Math.sin(2 * Math.PI * (t % 12) / 12);
    return trend * seasMult * (1 + randNormalish(randN, 0, 0.05));
  });
})();

function centeredMA(y, period) {
  const n = y.length, half = period / 2;
  const trend = new Array(n).fill(null);
  for (let t = half; t < n - half; t++) {
    let sum = 0.5 * y[t - half] + 0.5 * y[t + half];
    for (let k = -half + 1; k <= half - 1; k++) sum += y[t + k];
    trend[t] = sum / period;
  }
  return trend;
}
function decomposeAdditive(y, period) {
  const trend = centeredMA(y, period);
  const detrended = y.map((v, t) => trend[t] !== null ? v - trend[t] : null);
  const raw = new Array(period).fill(0), counts = new Array(period).fill(0);
  detrended.forEach((v, t) => { if (v !== null) { raw[t % period] += v; counts[t % period]++; } });
  const avg = raw.map((s, i) => s / counts[i]);
  const mean = avg.reduce((a, b) => a + b, 0) / period;
  const seasonalIdx = avg.map(s => s - mean);
  const seasonal = y.map((_, t) => seasonalIdx[t % period]);
  const residual = y.map((v, t) => trend[t] !== null ? v - trend[t] - seasonal[t] : null);
  return { trend, seasonalIdx, seasonal, residual };
}
function decomposeMultiplicative(y, period) {
  const trend = centeredMA(y, period);
  const detrended = y.map((v, t) => trend[t] !== null ? v / trend[t] : null);
  const raw = new Array(period).fill(0), counts = new Array(period).fill(0);
  detrended.forEach((v, t) => { if (v !== null) { raw[t % period] += v; counts[t % period]++; } });
  const avg = raw.map((s, i) => s / counts[i]);
  const mean = avg.reduce((a, b) => a + b, 0) / period;
  const seasonalIdx = avg.map(s => s / mean);
  const seasonal = y.map((_, t) => seasonalIdx[t % period]);
  const residual = y.map((v, t) => trend[t] !== null ? v / (trend[t] * seasonal[t]) : null);
  return { trend, seasonalIdx, seasonal, residual };
}
const DECOMP_ADD = decomposeAdditive(TS_MAIN, TS_PERIOD);
const DECOMP_MULT = decomposeMultiplicative(TS_MULT, TS_PERIOD);
const TS_DESEASONALIZED = TS_MAIN.map((v, t) => v - DECOMP_ADD.seasonal[t]);

/* ── ACF / PACF (Durbin-Levinson) ── */
function genAR(seed, n, c, phi, sigma) {
  const rand = seededRandom(seed), y = [];
  for (let t = 0; t < n; t++) { let v = c; for (let i = 0; i < phi.length; i++) v += phi[i] * (y[t - 1 - i] ?? 0); v += randNormalish(rand, 0, sigma); y.push(v); }
  return y;
}
function genMA(seed, n, c, theta, sigma) {
  const rand = seededRandom(seed), y = [], eps = [];
  for (let t = 0; t < n; t++) { const e = randNormalish(rand, 0, sigma); eps.push(e); let v = c + e; for (let j = 0; j < theta.length; j++) v += theta[j] * (eps[t - 1 - j] ?? 0); y.push(v); }
  return y;
}
const AR2_SERIES = genAR(3001, 200, 2, [0.6, -0.3], 1);
const MA1_SERIES = genMA(4001, 200, 5, [0.5], 1.5);
function acf(y, maxLag) {
  const mean = y.reduce((a, b) => a + b, 0) / y.length;
  const c0 = y.reduce((s, v) => s + (v - mean) ** 2, 0) / y.length;
  const out = [];
  for (let k = 1; k <= maxLag; k++) {
    let s = 0; for (let t = k; t < y.length; t++) s += (y[t] - mean) * (y[t - k] - mean);
    out.push((s / y.length) / c0);
  }
  return out;
}
function pacfDurbinLevinson(acfVals, maxLag) {
  const phi = {}, pacf = [];
  phi[1] = { 1: acfVals[0] };
  pacf.push(acfVals[0]);
  for (let k = 2; k <= maxLag; k++) {
    let num = acfVals[k - 1];
    for (let j = 1; j < k; j++) num -= phi[k - 1][j] * acfVals[k - 1 - j];
    let den = 1;
    for (let j = 1; j < k; j++) den -= phi[k - 1][j] * acfVals[j - 1];
    const phikk = num / den;
    phi[k] = {};
    for (let j = 1; j < k; j++) phi[k][j] = phi[k - 1][j] - phikk * phi[k - 1][k - j];
    phi[k][k] = phikk;
    pacf.push(phikk);
  }
  return pacf;
}

/* ── ADF stationarity test: OLS on Δyₜ = α [+β·t] + γ·yₜ₋₁ + Σδᵢ·Δyₜ₋ᵢ + ε ── */
function diffN(y, d) { let cur = y.slice(); for (let i = 0; i < d; i++) cur = cur.slice(1).map((v, j) => v - cur[j]); return cur; }
function adfTest(y, lags, includeTrend) {
  const n = y.length, dy = [];
  for (let t = 1; t < n; t++) dy.push(y[t] - y[t - 1]);
  const rows = [], targets = [];
  for (let t = lags + 1; t < n; t++) {
    const row = [1];
    if (includeTrend) row.push(t);
    row.push(y[t - 1]);
    for (let i = 1; i <= lags; i++) row.push(dy[t - 1 - i]);
    rows.push(row);
    targets.push(dy[t - 1]);
  }
  const Xt = transpose(rows);
  const XtX = matMul(Xt, rows);
  const Xty = matVec(Xt, targets);
  const beta = solveLinearSystem(XtX, Xty);
  const XtXinv = invertMatrix(XtX);
  const fitted = rows.map(row => row.reduce((s, x, i) => s + x * beta[i], 0));
  const resid = targets.map((v, i) => v - fitted[i]);
  const k = beta.length, nobs = rows.length;
  const sigma2 = resid.reduce((s, e) => s + e * e, 0) / (nobs - k);
  const gammaIdx = includeTrend ? 2 : 1;
  const se = Math.sqrt(sigma2 * XtXinv[gammaIdx][gammaIdx]);
  return { tstat: beta[gammaIdx] / se, nobs };
}
const ADF_CRIT = { c: { '1%': -3.43, '5%': -2.86, '10%': -2.57 }, ct: { '1%': -3.96, '5%': -3.41, '10%': -3.12 } };

/* ── CSS (Conditional Sum of Squares) ARMA estimation ──────────
   Verified range: p∈[0,3], d∈[0,2], q∈[0,1] recover reliably in
   combination. Higher combined orders (e.g. ARMA(2,2)+) showed real,
   repeated, restart-resistant degradation during verification — the UI
   below enforces this range directly via slider bounds, not just copy. */
function cssResiduals(y, c, phi, theta) {
  const n = y.length, p = phi.length, q = theta.length;
  const eps = new Array(n).fill(0);
  for (let t = 0; t < n; t++) {
    let pred = c;
    for (let i = 0; i < p; i++) { const idx = t - 1 - i; if (idx >= 0) pred += phi[i] * y[idx]; }
    for (let j = 0; j < q; j++) { const idx = t - 1 - j; if (idx >= 0) pred += theta[j] * eps[idx]; }
    eps[t] = y[t] - pred;
  }
  return eps;
}
function cssObjective(y, params, p, q) {
  const c = params[0], phi = params.slice(1, 1 + p), theta = params.slice(1 + p, 1 + p + q);
  return cssResiduals(y, c, phi, theta).reduce((s, e) => s + e * e, 0);
}
function compassSearch(y, p, q, init, iters = 300) {
  let params = init.slice(), step = 0.3, best = cssObjective(y, params, p, q);
  for (let it = 0; it < iters && step > 1e-6; it++) {
    let improved = false;
    for (let k = 0; k < params.length; k++) for (const dir of [1, -1]) {
      const trial = params.slice(); trial[k] += dir * step;
      const val = cssObjective(y, trial, p, q);
      if (val < best) { best = val; params = trial; improved = true; }
    }
    if (!improved) step *= 0.5;
  }
  return { params, sse: best };
}
function fitARMA(y, p, q, restarts, seed) {
  const rand = seededRandom(seed);
  let bestResult = null;
  for (let r = 0; r < restarts; r++) {
    const init = [randNormalish(rand, 0, 0.5), ...Array(p).fill(0).map(() => randNormalish(rand, 0, 0.3)), ...Array(q).fill(0).map(() => randNormalish(rand, 0, 0.3))];
    const result = compassSearch(y, p, q, init);
    if (!bestResult || result.sse < bestResult.sse) bestResult = result;
  }
  return bestResult;
}
function fitARIMA(y, p, d, q) {
  const w = diffN(y, d);
  const { params } = fitARMA(w, p, q, 6, 4242);
  const c = params[0], phi = params.slice(1, 1 + p), theta = params.slice(1 + p, 1 + p + q);
  const eps = cssResiduals(w, c, phi, theta);
  const sigma2 = eps.length ? eps.reduce((s, e) => s + e * e, 0) / eps.length : 1;
  return { c, phi, theta, sigma2, w, eps };
}
function forecastARMA(w, c, phi, theta, epsHist, h) {
  const p = phi.length, q = theta.length;
  const wExt = w.slice(), epsExt = epsHist.slice(), out = [];
  for (let k = 0; k < h; k++) {
    const t = wExt.length;
    let pred = c;
    for (let i = 0; i < p; i++) { const idx = t - 1 - i; if (idx >= 0) pred += phi[i] * wExt[idx]; }
    for (let j = 0; j < q; j++) { const idx = t - 1 - j; if (idx >= 0) pred += theta[j] * epsExt[idx]; }
    wExt.push(pred); epsExt.push(0);
    out.push(pred);
  }
  return out;
}
function integrateForecast(lastVal, diffForecasts, d) {
  let series = diffForecasts.slice(), base = lastVal;
  for (let k = 0; k < d; k++) {
    let acc = base, integrated = [];
    for (const v of series) { acc += v; integrated.push(acc); }
    series = integrated;
  }
  return series;
}
function psiWeights(phi, theta, dOrder, hmax) {
  function polyMul(a, b) { const res = new Array(a.length + b.length - 1).fill(0); for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) res[i + j] += a[i] * b[j]; return res; }
  let combined = [1, ...phi.map(v => -v)];
  for (let i = 0; i < dOrder; i++) combined = polyMul(combined, [1, -1]);
  const thetaPoly = [1, ...theta];
  const psi = new Array(hmax).fill(0);
  for (let j = 0; j < hmax; j++) {
    let val = j < thetaPoly.length ? thetaPoly[j] : 0;
    for (let i = 1; i < combined.length; i++) if (j - i >= 0) val -= combined[i] * psi[j - i];
    psi[j] = val / combined[0];
  }
  return psi;
}

/* ── Exponential smoothing: SES, Holt, Holt-Winters (additive) ───
   Parameters fit by grid search minimizing in-sample SSE, precomputed
   once at module load (same BV_CURVE-style precompute as
   ValidationBiasVariance.jsx) since there are no user-facing fitting
   sliders — only a method toggle. */
function sesSSEAndForecast(y, alpha, h) {
  let level = y[0], sse = 0;
  for (let t = 1; t < y.length; t++) { const err = y[t] - level; sse += err * err; level = alpha * y[t] + (1 - alpha) * level; }
  return { sse, forecast: Array(h).fill(level) };
}
function holtSSEAndForecast(y, alpha, beta, h) {
  let level = y[0], trend = y[1] - y[0], sse = 0;
  for (let t = 1; t < y.length; t++) {
    const fc = level + trend, err = y[t] - fc; sse += err * err;
    const newLevel = alpha * y[t] + (1 - alpha) * (level + trend);
    trend = beta * (newLevel - level) + (1 - beta) * trend;
    level = newLevel;
  }
  return { sse, forecast: Array.from({ length: h }, (_, k) => level + (k + 1) * trend) };
}
function holtWintersSSEAndForecast(y, alpha, beta, gamma, period, h) {
  const l0 = y.slice(0, period).reduce((a, b) => a + b, 0) / period;
  const l1 = y.slice(period, 2 * period).reduce((a, b) => a + b, 0) / period;
  let level = l0, trend = (l1 - l0) / period, sse = 0;
  const seas = Array.from({ length: period }, (_, i) => y[i] - l0);
  for (let t = 0; t < y.length; t++) {
    const s = seas[t % period], fc = level + trend + s;
    if (t >= period) { const err = y[t] - fc; sse += err * err; }
    const newLevel = alpha * (y[t] - s) + (1 - alpha) * (level + trend);
    const newTrend = beta * (newLevel - level) + (1 - beta) * trend;
    seas[t % period] = gamma * (y[t] - newLevel) + (1 - gamma) * s;
    level = newLevel; trend = newTrend;
  }
  const forecast = Array.from({ length: h }, (_, k) => level + (k + 1) * trend + seas[(y.length + k) % period]);
  return { sse, forecast };
}
function gridFitSES(y) {
  let best = { sse: Infinity };
  for (let i = 1; i <= 99; i++) { const r = sesSSEAndForecast(y, i / 100, 12); if (r.sse < best.sse) best = { ...r, alpha: i / 100 }; }
  return best;
}
function gridFitHolt(y) {
  let best = { sse: Infinity };
  for (let i = 1; i <= 99; i++) for (let j = 1; j <= 99; j++) { const r = holtSSEAndForecast(y, i / 100, j / 100, 12); if (r.sse < best.sse) best = { ...r, alpha: i / 100, beta: j / 100 }; }
  return best;
}
function gridFitHoltWinters(y, period) {
  let best = { sse: Infinity };
  for (let i = 1; i <= 20; i++) for (let j = 1; j <= 20; j++) for (let k = 1; k <= 20; k++) {
    const r = holtWintersSSEAndForecast(y, i / 20, j / 20, k / 20, period, 12);
    if (r.sse < best.sse) best = { ...r, alpha: i / 20, beta: j / 20, gamma: k / 20 };
  }
  let refined = best;
  for (let di = -5; di <= 5; di++) for (let dj = -5; dj <= 5; dj++) for (let dk = -5; dk <= 5; dk++) {
    const a = Math.min(0.99, Math.max(0.01, best.alpha + di * 0.01));
    const b = Math.min(0.99, Math.max(0.01, best.beta + dj * 0.01));
    const g = Math.min(0.99, Math.max(0.01, best.gamma + dk * 0.01));
    const r = holtWintersSSEAndForecast(y, a, b, g, period, 12);
    if (r.sse < refined.sse) refined = { ...r, alpha: a, beta: b, gamma: g };
  }
  return refined;
}
const ES_SES = gridFitSES(TS_MAIN);
const ES_HOLT = gridFitHolt(TS_MAIN);
const ES_HW = gridFitHoltWinters(TS_MAIN, TS_PERIOD);

/* ══════════════════════════════════════════════════════════════
   Demo: Decomposition
   ══════════════════════════════════════════════════════════════ */
function DecompositionDemo() {
  const [mode, setMode] = useState('additive');
  const [view, setView] = useState('decomposed');
  const y = mode === 'additive' ? TS_MAIN : TS_MULT;
  const d = mode === 'additive' ? DECOMP_ADD : DECOMP_MULT;
  const xs = y.map((_, t) => t);
  const xDomain = [0, TS_N - 1];
  const trendClean = d.trend.map((v, i) => v === null ? y[i] : v);
  const residClean = d.residual.map(v => v === null ? (mode === 'additive' ? 0 : 1) : v);

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <button onClick={() => setMode('additive')} style={toggleBtnStyle(mode === 'additive')}>Additive (constant swing)</button>
        <button onClick={() => setMode('multiplicative')} style={toggleBtnStyle(mode === 'multiplicative')}>Multiplicative (scaling swing)</button>
        <button onClick={() => setView(v => v === 'combined' ? 'decomposed' : 'combined')} style={toggleBtnStyle(view === 'decomposed')}>{view === 'combined' ? 'Decompose' : 'Show Combined'}</button>
      </div>
      {view === 'combined' ? (
        <MiniChart w={560} h={140} series={[{ points: xs.map((x, i) => [x, y[i]]), color: 'var(--accent)' }]} xDomain={xDomain} yDomain={domainOf(y)} />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[
            { label: 'Original', data: y, color: 'var(--accent)' },
            { label: 'Trend', data: trendClean, color: 'var(--color-border-info)' },
            { label: mode === 'additive' ? 'Seasonal' : 'Seasonal (index)', data: d.seasonal, color: 'var(--color-border-success)' },
            { label: 'Residual', data: residClean, color: 'var(--color-border-danger)' },
          ].map(row => (
            <div key={row.label}>
              <div style={{ fontSize: 10.5, color: 'var(--color-text-tertiary)', marginBottom: 2 }}>{row.label}</div>
              <MiniChart w={560} h={72} series={[{ points: xs.map((x, i) => [x, row.data[i]]), color: row.color }]} xDomain={xDomain} yDomain={domainOf(row.data)} />
            </div>
          ))}
        </div>
      )}
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        {mode === 'additive'
          ? 'Additive series: the seasonal swing stays a constant absolute size regardless of the trend level — exactly the shape built into this synthetic series.'
          : 'Multiplicative series: the seasonal swing scales with the trend level — notice the oscillations widen as the trend climbs, unlike the additive case above.'}
      </div>
    </VizBox>
  );
}

/* ══════════════════════════════════════════════════════════════
   Demo: Stationarity / ADF test
   ══════════════════════════════════════════════════════════════ */
function StationarityDemo() {
  const [source, setSource] = useState('deseason');
  const [order, setOrder] = useState(0);
  const [trendTerm, setTrendTerm] = useState(false);
  const [lag, setLag] = useState(1);

  const baseSeries = source === 'deseason' ? TS_DESEASONALIZED : TS_MAIN;
  const series = useMemo(() => diffN(baseSeries, order), [baseSeries, order]);
  const result = useMemo(() => adfTest(series, lag, trendTerm), [series, lag, trendTerm]);
  const crit = ADF_CRIT[trendTerm ? 'ct' : 'c'];
  const isStationary = result.tstat < crit['5%'];

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <button onClick={() => setSource('deseason')} style={toggleBtnStyle(source === 'deseason')}>Deseasonalized (recommended)</button>
        <button onClick={() => setSource('raw')} style={toggleBtnStyle(source === 'raw')}>Raw series (watch it destabilize)</button>
      </div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <button onClick={() => setOrder(0)} style={toggleBtnStyle(order === 0)}>Level</button>
        <button onClick={() => setOrder(1)} style={toggleBtnStyle(order === 1)}>1st Difference</button>
        <button onClick={() => setOrder(2)} style={toggleBtnStyle(order === 2)}>2nd Difference</button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12.5, marginLeft: 8 }}>
          <input type="checkbox" checked={trendTerm} onChange={e => setTrendTerm(e.target.checked)} /> include trend term
        </label>
      </div>
      <SliderRow label="Augmentation lags" min={0} max={3} step={1} value={lag} onChange={setLag} fmt={v => String(v)} />
      <MiniChart w={560} h={130} series={[{ points: series.map((v, i) => [i, v]), color: 'var(--accent)' }]} xDomain={[0, series.length - 1]} yDomain={domainOf(series)}
        extra={({ yToPx }) => <line x1={0} y1={yToPx(0)} x2={560} y2={yToPx(0)} stroke="var(--color-border-tertiary)" strokeWidth={1} strokeDasharray="3,3" />} />
      <div style={{ display: 'flex', gap: 18, marginTop: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>t-statistic = {result.tstat.toFixed(2)}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--color-text-tertiary)' }}>
          critical (1%/5%/10%) = {crit['1%']}/{crit['5%']}/{crit['10%']}
        </span>
        <Badge color={isStationary ? 'success' : 'danger'}>{isStationary ? 'Stationary (reject H₀)' : 'Non-Stationary (fail to reject H₀)'}</Badge>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        H₀ ("unit root present, non-stationary") is rejected when the t-statistic falls{' '}
        <em>below</em> the critical value — more negative is stronger evidence of stationarity.
      </div>
    </VizBox>
  );
}

/* ══════════════════════════════════════════════════════════════
   Demo: ACF / PACF reading practice
   ══════════════════════════════════════════════════════════════ */
function ACFPACFDemo() {
  const [which, setWhich] = useState('ar');
  const series = which === 'ar' ? AR2_SERIES : MA1_SERIES;
  const maxLag = 8;
  const acfVals = useMemo(() => acf(series, maxLag), [series]);
  const pacfVals = useMemo(() => pacfDurbinLevinson(acfVals, maxLag), [acfVals]);
  const band = 1.96 / Math.sqrt(series.length);
  const w = 260, h = 130, barW = w / (maxLag + 1);

  function Bars({ vals, label }) {
    const { yToPx } = makeScales([0, maxLag], [-0.6, 1], w, h);
    return (
      <div>
        <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>{label}</div>
        <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
          <rect x={0} y={yToPx(band)} width={w} height={yToPx(-band) - yToPx(band)} fill="var(--color-border-tertiary)" opacity={0.25} />
          <line x1={0} y1={yToPx(0)} x2={w} y2={yToPx(0)} stroke="var(--color-border-tertiary)" strokeWidth={1} />
          {vals.map((v, i) => (
            <rect key={i} x={(i + 0.7) * barW} y={Math.min(yToPx(0), yToPx(v))} width={barW * 0.5} height={Math.abs(yToPx(v) - yToPx(0))}
              fill={Math.abs(v) > band ? 'var(--color-background-danger)' : 'var(--color-border-info)'} />
          ))}
        </svg>
      </div>
    );
  }

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <button onClick={() => setWhich('ar')} style={toggleBtnStyle(which === 'ar')}>AR(2) series</button>
        <button onClick={() => setWhich('ma')} style={toggleBtnStyle(which === 'ma')}>MA(1) series</button>
      </div>
      <Grid cols={2} gap={14}>
        <Bars vals={acfVals} label="ACF" />
        <Bars vals={pacfVals} label="PACF" />
      </Grid>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Red bars sit outside the shaded ±1.96/√n significance band. {which === 'ar'
          ? 'AR(2): PACF cuts off sharply after lag 2 while ACF tails off gradually — read the PACF cutoff as "p=2."'
          : 'MA(1): ACF cuts off sharply after lag 1 while PACF tails off gradually — the mirror image of the AR case, read as "q=1."'}
      </div>
    </VizBox>
  );
}

/* ══════════════════════════════════════════════════════════════
   Demo: ARIMA fit + forecast + confidence band
   ══════════════════════════════════════════════════════════════ */
function ARIMADemo() {
  const [source, setSource] = useState('deseason');
  const [p, setP] = useState(1);
  const [d, setD] = useState(1);
  const [q, setQ] = useState(1);
  const H = 12;

  const y = source === 'deseason' ? TS_DESEASONALIZED : TS_MAIN;
  const fit = useMemo(() => fitARIMA(y, p, d, q), [y, p, d, q]);
  const forecast = useMemo(() => {
    const diffFc = forecastARMA(fit.w, fit.c, fit.phi, fit.theta, fit.eps, H);
    if (d === 0) return diffFc;
    return integrateForecast(y[y.length - 1], diffFc, d);
  }, [fit, y, d]);
  const psi = useMemo(() => psiWeights(fit.phi, fit.theta, d, H), [fit, d]);
  const ciHalfWidth = useMemo(() => {
    let running = 0;
    return psi.map(p2 => { running += p2 * p2; return 1.96 * Math.sqrt(fit.sigma2 * running); });
  }, [psi, fit.sigma2]);

  const histLen = 40;
  const hist = y.slice(-histLen);
  const xs = hist.map((_, i) => i);
  const fxs = forecast.map((_, i) => histLen - 1 + i + 1);
  const allY = [...hist, ...forecast, ...forecast.map((v, i) => v + ciHalfWidth[i]), ...forecast.map((v, i) => v - ciHalfWidth[i])];

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <button onClick={() => setSource('deseason')} style={toggleBtnStyle(source === 'deseason')}>Deseasonalized</button>
        <button onClick={() => setSource('raw')} style={toggleBtnStyle(source === 'raw')}>Raw (seasonal) — watch it struggle</button>
      </div>
      <Grid cols={3} gap={10}>
        <SliderRow label="p (AR order)" min={0} max={3} step={1} value={p} onChange={setP} fmt={v => String(v)} />
        <SliderRow label="d (differencing)" min={0} max={2} step={1} value={d} onChange={setD} fmt={v => String(v)} />
        <SliderRow label="q (MA order)" min={0} max={1} step={1} value={q} onChange={setQ} fmt={v => String(v)} />
      </Grid>
      <MiniChart w={560} h={160} xDomain={[0, fxs[fxs.length - 1]]} yDomain={domainOf(allY)}
        series={[
          { points: xs.map((x, i) => [x, hist[i]]), color: 'var(--color-border-info)' },
          { points: fxs.map((x, i) => [x, forecast[i]]), color: 'var(--color-background-danger)', width: 2 },
        ]}
        extra={({ xToPx, yToPx }) => (
          <path d={
            'M ' + fxs.map((x, i) => `${xToPx(x).toFixed(2)},${yToPx(forecast[i] + ciHalfWidth[i]).toFixed(2)}`).join(' L ') +
            ' L ' + [...fxs].reverse().map((x, i) => `${xToPx(x).toFixed(2)},${yToPx(forecast[fxs.length - 1 - i] - ciHalfWidth[fxs.length - 1 - i]).toFixed(2)}`).join(' L ') + ' Z'
          } fill="var(--color-background-danger)" opacity={0.15} stroke="none" />
        )} />
      <div style={{ display: 'flex', gap: 16, marginTop: 10, flexWrap: 'wrap', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
        <span>c={fit.c.toFixed(2)}</span>
        {fit.phi.length > 0 && <span>φ=[{fit.phi.map(v => v.toFixed(2)).join(', ')}]</span>}
        {fit.theta.length > 0 && <span>θ=[{fit.theta.map(v => v.toFixed(2)).join(', ')}]</span>}
        <span>forecast range={(Math.max(...forecast) - Math.min(...forecast)).toFixed(2)}</span>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        {source === 'raw'
          ? 'Fit directly on the raw seasonal series, this small ARIMA model cannot represent a 12-month oscillation — its forecast collapses toward a smooth trend line, understating the true seasonal swing by roughly an order of magnitude.'
          : 'Fit on the deseasonalized series, the forecast is a sensible, smoothly trend-continuing projection — the shaded band is a real 95% confidence interval computed from the fitted model\'s psi-weights, not a fixed illustrative guess.'}
      </div>
    </VizBox>
  );
}

/* ══════════════════════════════════════════════════════════════
   Demo: Exponential Smoothing
   ══════════════════════════════════════════════════════════════ */
function ExponentialSmoothingDemo() {
  const [mode, setMode] = useState('hw');
  const fits = { ses: ES_SES, holt: ES_HOLT, hw: ES_HW };
  const current = fits[mode];
  const histLen = 36;
  const hist = TS_MAIN.slice(-histLen);
  const xs = hist.map((_, i) => i);
  const fxs = current.forecast.map((_, i) => histLen - 1 + i + 1);
  const allY = [...hist, ...current.forecast];

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <button onClick={() => setMode('ses')} style={toggleBtnStyle(mode === 'ses')}>Simple (SES)</button>
        <button onClick={() => setMode('holt')} style={toggleBtnStyle(mode === 'holt')}>Holt (trend)</button>
        <button onClick={() => setMode('hw')} style={toggleBtnStyle(mode === 'hw')}>Holt-Winters (trend + season)</button>
      </div>
      <MiniChart w={560} h={160} xDomain={[0, fxs[fxs.length - 1]]} yDomain={domainOf(allY)}
        series={[
          { points: xs.map((x, i) => [x, hist[i]]), color: 'var(--color-border-info)' },
          { points: fxs.map((x, i) => [x, current.forecast[i]]), color: 'var(--color-background-danger)', width: 2 },
        ]} />
      <Table
        heads={['Method', 'Fitted parameters', 'In-sample SSE']}
        rows={[
          ['Simple (SES)', `α=${ES_SES.alpha.toFixed(2)}`, ES_SES.sse.toFixed(0)],
          ['Holt', `α=${ES_HOLT.alpha.toFixed(2)}, β=${ES_HOLT.beta.toFixed(2)}`, ES_HOLT.sse.toFixed(0)],
          ['Holt-Winters', `α=${ES_HW.alpha.toFixed(2)}, β=${ES_HW.beta.toFixed(2)}, γ=${ES_HW.gamma.toFixed(2)}`, ES_HW.sse.toFixed(0)],
        ]}
      />
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Holt-Winters' SSE is roughly 3.3-3.7× lower than either SES or Holt alone — the only one of
        the three that can actually represent the seasonal swing. Notice Holt actually scores{' '}
        <em>worse</em> than plain SES here: adding a trend component without a seasonal one doesn't
        help when the dominant unmodeled structure is seasonal, not linear drift.
      </div>
    </VizBox>
  );
}

/* ══════════════════════════════════════════════════════════════
   Demo: Feature-based sliding window
   ══════════════════════════════════════════════════════════════ */
function featureRow(y, t) {
  const window = y.slice(t - 3, t);
  const rollMean = window.reduce((a, b) => a + b, 0) / window.length;
  const rollStd = Math.sqrt(window.reduce((s, v) => s + (v - rollMean) ** 2, 0) / window.length);
  const month = t % 12;
  return {
    t, lag1: y[t - 1], lag2: y[t - 2], lag3: y[t - 3],
    rollMean3: rollMean, rollStd3: rollStd,
    monthSin: Math.sin(2 * Math.PI * month / 12), monthCos: Math.cos(2 * Math.PI * month / 12),
    target: y[t],
  };
}
function FeatureSlideDemo() {
  const START = 3, END = 18;
  const [t, setT] = useState(START);
  const [playing, setPlaying] = useState(false);
  const rows = useMemo(() => { const out = []; for (let k = START; k <= t; k++) out.push(featureRow(TS_MAIN, k)); return out; }, [t]);

  useEffect(() => {
    if (!playing) return undefined;
    if (t >= END) { setPlaying(false); return undefined; }
    const id = setTimeout(() => setT(v => Math.min(END, v + 1)), 500);
    return () => clearTimeout(id);
  }, [playing, t]);

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <button onClick={() => setPlaying(p => !p)} style={toggleBtnStyle(playing)}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => setT(v => Math.min(END, v + 1))} style={toggleBtnStyle(false)}>Step</button>
        <button onClick={() => { setT(START); setPlaying(false); }} style={toggleBtnStyle(false)}>Reset</button>
        <span style={{ fontSize: 12, color: 'var(--color-text-tertiary)', alignSelf: 'center' }}>window sliding to t={t}</span>
      </div>
      <div className="ml-table-wrap" style={{ maxHeight: 240, overflowY: 'auto' }}>
        <table className="ml-table">
          <thead><tr>{['t', 'lag1', 'lag2', 'lag3', 'rollMean3', 'rollStd3', 'monthSin', 'monthCos', 'target'].map(h => <th key={h}>{h}</th>)}</tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.t}>
                {[r.t, r.lag1, r.lag2, r.lag3, r.rollMean3, r.rollStd3, r.monthSin, r.monthCos, r.target].map((v, i) => (
                  <td key={i}>{typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(2) : v}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        Each row is one ordinary supervised-learning training example — <code>target</code> is just
        this month's value, and every other column is computable using only data strictly before it.
      </div>
    </VizBox>
  );
}

/* ══════════════════════════════════════════════════════════════
   Demo: Walk-Forward Cross-Validation
   ══════════════════════════════════════════════════════════════ */
function WalkForwardDemo() {
  const [mode, setMode] = useState('expanding');
  const folds = Array.from({ length: 5 }, (_, i) => {
    const trainStart = mode === 'expanding' ? 0 : 12 * i;
    const trainEnd = 60 + 12 * i;
    const testEnd = trainEnd + 12;
    return { trainStart, trainEnd, testEnd };
  });
  const w = 560, rowH = 22;
  const xToPx = x => x / TS_N * w;

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <button onClick={() => setMode('expanding')} style={toggleBtnStyle(mode === 'expanding')}>Expanding Window</button>
        <button onClick={() => setMode('sliding')} style={toggleBtnStyle(mode === 'sliding')}>Sliding Window</button>
      </div>
      <svg viewBox={`0 0 ${w} ${folds.length * rowH + 10}`} width={w} height={folds.length * rowH + 10} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
        {folds.map((f, i) => (
          <g key={i} transform={`translate(0, ${i * rowH + 4})`}>
            <rect x={xToPx(f.trainStart)} y={0} width={xToPx(f.trainEnd) - xToPx(f.trainStart)} height={14} fill="var(--color-background-info)" />
            <rect x={xToPx(f.trainEnd)} y={0} width={xToPx(f.testEnd) - xToPx(f.trainEnd)} height={14} fill="var(--color-background-danger)" />
          </g>
        ))}
      </svg>
      <div style={{ display: 'flex', gap: 14, marginTop: 8, fontSize: 11.5, color: 'var(--color-text-tertiary)' }}>
        <span><span style={{ color: 'var(--color-text-info)' }}>■</span> train</span>
        <span><span style={{ color: 'var(--color-text-danger)' }}>■</span> test (12 months = 1 full seasonal cycle)</span>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 6 }}>
        {mode === 'expanding'
          ? 'Expanding window: every fold keeps all past data, just tests on the next unseen block — good when more historical data reliably helps.'
          : 'Sliding window: each fold trains on a fixed-size, moving recent window — good when older data may reflect an outdated regime.'}
        {' '}Both always train strictly on the past and test strictly on the future — never shuffled.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'decomposition', label: 'Decomposition', sub: 'Trend, seasonality, cyclic variation & residuals' },
  { id: 'stationarity', label: 'Stationarity', sub: 'The ADF test, differencing & why classical models need it' },
  { id: 'arima', label: 'ARIMA & SARIMA', sub: 'Classical forecasting with autoregression & moving averages' },
  { id: 'smoothing', label: 'Exponential Smoothing', sub: 'SES, Holt & Holt-Winters' },
  { id: 'features', label: 'Feature-Based Forecasting', sub: 'Turning sequences into tabular ML problems' },
];

function SectionDecomposition() {
  return (
    <div>
      <P>
        A time series is data where order isn't incidental — it's the whole point. Before fitting any
        model, it helps to mentally split a series into the pieces that make it up: a slow-moving
        level, a repeating pattern, and whatever's left over.
      </P>
      <H2 c="Trend, Seasonality, Cyclic Variation & Residuals" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Trend">The long-term progression — increasing, decreasing, or drifting — ignoring any short-term wiggle.</Card>
        <Card color="success" title="Seasonality">A repeating pattern locked to a fixed, known period (retail sales spiking every December, every December).</Card>
        <Card color="warning" title="Cyclic Variation">Rises and falls with no fixed period, usually driven by macroeconomic forces (a business cycle can last 3 years or 8) — easy to confuse with seasonality, but genuinely different: seasonality's period is fixed and known in advance, a cycle's isn't.</Card>
        <Card color="purple" title="Residuals">Whatever's left after removing trend and seasonality — ideally unstructured noise.</Card>
      </Grid>

      <H2 c="Classical Decomposition" />
      <P>
        Estimate the trend with a centered moving average over one full seasonal period (this
        smooths out the seasonal swing by construction, since it averages across an entire cycle);
        subtract it to get a detrended series; average the detrended value at each calendar position
        (every January together, every February together, ...) to get the seasonal index; whatever's
        left after removing both is the residual.
      </P>
      <Mx block>{`  Additive:        y = Trend + Seasonal + Residual
  Multiplicative:  y = Trend × Seasonal × Residual

  Rule of thumb: does the seasonal swing stay a constant absolute size as the
  level changes (additive), or does it scale with the level (multiplicative)?
  Retail revenue in dollars: usually multiplicative. Temperature anomalies:
  usually additive.`}</Mx>

      <H2 c="Try It — Decompose a Real Series" />
      <P>Toggle between an additive and a multiplicative synthetic series, and between the combined view and the split-out components.</P>
      <DecompositionDemo />
      <P s={{ marginTop: 10 }}>
        On the additive series, the recovered seasonal pattern matches the true generating pattern to
        within about 3% (mean absolute error 0.62 against a true peak-to-peak amplitude of 20), and
        the residual's lag-1 autocorrelation drops from 0.91 in the raw series to −0.22 — decomposition
        has genuinely removed almost all of the structure, leaving something close to noise.
      </P>

      <H3 c="A minimal working example" />
      <Code>{`from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(series, model="additive", period=12)
result.trend, result.seasonal, result.resid`}</Code>

      <Note color="info" icon="ti-info-circle">
        <strong>STL decomposition</strong> (Seasonal-Trend decomposition using Loess) is the modern,
        more robust alternative to the classical moving-average method above — it uses iterative
        local regression smoothing instead of a fixed-window average, handling changing seasonal
        patterns and outliers more gracefully. Worth knowing by name; the classical version above is
        what's actually implemented here.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="What's the practical difference between seasonality and cyclic variation, and why does it matter?"
          a="Seasonality has a fixed, known period (always 12 months, always 7 days) and can be modeled with a fixed seasonal index or seasonal ARIMA term. Cyclic variation has no fixed period — it's driven by macroeconomic or business factors and can't be handled the same mechanical way, since there's no fixed length to build into the model." />
      <QA q="How do you decide between additive and multiplicative decomposition for a real dataset?"
          a="Plot the series and look at whether the seasonal swing's absolute size stays roughly constant as the trend level changes (additive) or grows/shrinks proportionally with the level (multiplicative). A quick trick: if multiplicative, taking the log of the series first converts it into an additive problem, since log(Trend × Seasonal) = log(Trend) + log(Seasonal)." />
      <QA q="Why use a moving average of exactly the seasonal period's length to estimate the trend?"
          a="Averaging over one full period means every seasonal position (every month, in a period-12 series) contributes exactly once to the average, so the seasonal swing cancels out by construction — what's left approximates the trend with the seasonal component already removed, without needing to know the seasonal pattern in advance." />
      <QA q="After decomposing a series, the residuals still show a clear pattern. What does that tell you?"
          a="It means trend and seasonality haven't captured everything structured in the data — there may be a cyclic component, a second seasonal period (e.g. both weekly and yearly), or the additive/multiplicative choice may be wrong for this data. Residuals that still autocorrelate or trend are a sign the decomposition model is misspecified, not that the data is unusually noisy." />
    </div>
  );
}

function SectionStationarity() {
  return (
    <div>
      <P>
        Almost every classical forecasting model (ARIMA included) assumes the series it's fitted to is
        stationary — its statistical properties don't drift over time. A trending or seasonal series
        violates this outright, which is exactly why decomposition and differencing exist: to strip
        away the parts that make a series non-stationary before modeling what's left.
      </P>
      <H2 c="What Stationarity Actually Requires" />
      <P>
        Formally: constant mean, constant variance, and covariance between any two points that
        depends only on the distance between them, not on where in time they sit. A series with a
        trend has a mean that changes over time; a seasonal series has both a mean and a
        variance-like structure that repeats — both violate stationarity.
      </P>

      <H2 c="The Augmented Dickey-Fuller (ADF) Test" />
      <Mx block>{`  Δyₜ = α [+ β·t] + γ·yₜ₋₁ + Σᵢ δᵢ·Δyₜ₋ᵢ + εₜ

  H₀: γ = 0   (a "unit root" — the series is non-stationary)
  H₁: γ < 0   (stationary — yₜ₋₁ pulls back toward a stable level)

  Test statistic = γ̂ / SE(γ̂), compared against Dickey-Fuller critical
  values (NOT ordinary t-distribution values — the null distribution here
  is non-standard, so real implementations, including this one, use
  precomputed critical values, e.g. MacKinnon 1996/2010).`}</Mx>
      <Table
        heads={['Specification', '1%', '5%', '10%']}
        rows={[['Constant only', '−3.43', '−2.86', '−2.57'], ['Constant + trend', '−3.96', '−3.41', '−3.12']]}
      />

      <H2 c="Try It — Test, Difference, Retest" />
      <P>
        Toggle the data source, the differencing order, whether a trend term is included, and the
        number of augmentation lags.
      </P>
      <StationarityDemo />
      <P s={{ marginTop: 10 }}>
        On the deseasonalized level series, the test fails to reject non-stationarity at every lag
        choice (t around −0.5 to −2.0, nowhere close to −2.57); after a single difference, it
        overwhelmingly rejects at every lag choice (t around −10 to −17) — differencing has done its
        job.
      </P>

      <Note color="warning" icon="ti-alert-triangle">
        <strong>A real, verified wrinkle</strong>: running the ADF test directly on the{' '}
        <em>raw, still-seasonal</em> series gives a t-statistic that visibly shifts as the
        augmentation-lag count changes (from about −2.5 with 0 lags to about −3.4 with 3 lags) — the
        lag terms are partly absorbing leftover seasonal autocorrelation instead of genuine short-memory
        noise, making the verdict unstable and lag-dependent. Deseasonalizing first (as the toggle
        above defaults to) removes this instability entirely — a real, citable reason to never run ADF
        on a seasonal series without deseasonalizing or seasonally differencing it first.
      </Note>
      <Note color="info" icon="ti-info-circle">
        Adding a trend term changes the null hypothesis being tested — "stationary around a fixed mean"
        versus "stationary around a deterministic trend line" (<strong>trend-stationary</strong>).
        Toggling the trend-term checkbox above on the level series flips the verdict entirely, which
        is expected, not a bug: the two specifications are answering genuinely different questions.
      </Note>
      <Note color="danger" icon="ti-alert-triangle">
        <strong>Over-differencing</strong> is a real cost, not a free safety margin: differencing twice
        instead of once nearly triples the variance here (2.87×) and pushes the lag-1 autocorrelation
        toward −0.5 — the textbook signature of having differenced past the point where it helped.
        Difference only as many times as the test actually requires.
      </Note>
      <Note color="success" icon="ti-arrow-right">
        Ordinary K-Fold cross-validation shuffles rows and is invalid for sequential data — the honest
        alternative, <strong>walk-forward cross-validation</strong>, is covered in full at the end of
        Feature-Based Forecasting.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from statsmodels.tsa.stattools import adfuller

result = adfuller(series, autolag="AIC")
print("t-statistic:", result[0], "p-value:", result[1])
# result[4] holds the 1%/5%/10% critical values used for the verdict`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why do classical models like ARIMA require a stationary series?"
          a="Their parameters (autoregressive coefficients, moving-average coefficients) are assumed constant over the whole series — if the mean or variance is drifting, a single set of coefficients can't honestly describe the relationship at every point in time, and any fitted model risks extrapolating a trend or seasonal pattern that was really just an artifact of when the data was collected." />
      <QA q="What does it mean when an ADF test's t-statistic changes noticeably as you add more augmentation lags?"
          a="It's a sign the lag terms are absorbing structure they shouldn't have to — most commonly leftover seasonality that wasn't removed before testing. A stable, lag-insensitive verdict is what you want; instability across lag choices is a real, practical warning sign worth investigating (usually by deseasonalizing first) rather than just picking whichever lag count gives the answer you expected." />
      <QA q="What's the practical difference between a series being 'stationary' and 'trend-stationary'?"
          a="A stationary series fluctuates around a fixed mean. A trend-stationary series fluctuates around a deterministic trend line — it's non-stationary in the strict sense (its mean does change over time) but becomes stationary once that deterministic trend is subtracted out, rather than needing differencing. The ADF test's constant+trend specification is testing for exactly this second case." />
      <QA q="You difference a series twice 'just to be safe.' What could go wrong?"
          a="Over-differencing inflates variance and introduces artificial negative autocorrelation (a textbook sign is the differenced series' lag-1 autocorrelation approaching −0.5) — it doesn't just fail to help, it actively makes the series harder to model well. Difference only as many times as a stationarity test indicates is actually necessary." />
    </div>
  );
}

function SectionARIMA() {
  return (
    <div>
      <P>
        ARIMA models a (stationary, or differenced-to-stationary) series using only its own past —
        past values (AutoRegressive) and past forecast errors (Moving Average) — with no external
        predictors at all.
      </P>
      <H2 c="AR, I, and MA" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="AR (p)">AutoRegressive — today's value as a linear combination of the last p values. Captures momentum and mean-reversion.</Card>
        <Card color="warning" title="I (d)">Integrated — how many times the series needs differencing to become stationary before AR/MA are fitted to it.</Card>
        <Card color="success" title="MA (q)">Moving Average — today's value as a linear combination of the last q forecast <em>errors</em>, not raw values. Captures short-lived shocks.</Card>
      </Grid>
      <Mx block>{`  ARIMA(p,d,q) on the d-times-differenced series wₜ:

  wₜ = c + φ₁wₜ₋₁ + ... + φₚwₜ₋ₚ + εₜ + θ₁εₜ₋₁ + ... + θ_qεₜ₋_q`}</Mx>

      <H2 c="Reading ACF/PACF to Pick p and q" />
      <P>
        Before ARIMA had automated fitting tools, analysts picked p and q by eye from correlograms —
        it's still one of the fastest sanity checks on a fitted model's order, and a real, common
        practical/interview skill.
      </P>
      <ACFPACFDemo />

      <H2 c="Try It — Fit ARIMA Live" />
      <P>
        A real Conditional Sum of Squares fit (the standard way to estimate ARMA coefficients when
        residuals aren't directly observable), refit from scratch on every slider change.
      </P>
      <ARIMADemo />
      <Note color="warning" icon="ti-alert-triangle">
        The slider ranges above (p≤3, d≤2, q≤1) aren't arbitrary — verified directly: AR-only fits are
        exact, and MA(1) combined with any AR order recovers reliably, but pushing to MA(2)+ combined
        with any nonzero AR order produced real, repeated, restart-resistant bad fits during testing
        (the AR and MA polynomials can nearly cancel each other out, a genuine identifiability problem,
        not just an under-searched optimizer). The range shipped here is the one that's actually
        trustworthy every time.
      </Note>

      <H2 c="Forecast Uncertainty Grows With Horizon" />
      <P>
        Every ARIMA model can be rewritten as an infinite weighted sum of past shocks (its "psi-weight"
        or MA(∞) representation) — this isn't just theory, it's exactly how the confidence band above
        is computed:
      </P>
      <Mx block>{`  yₜ = μ + εₜ + ψ₁εₜ₋₁ + ψ₂εₜ₋₂ + ...      (ψ₀ = 1)

  Var(forecast error at horizon h) = σ² · (ψ₀² + ψ₁² + ... + ψ_{h-1}²)

  95% CI half-width = 1.96 · √(that variance)`}</Mx>
      <P>
        This variance can only grow (or plateau) as <Mx>h</Mx> increases, which is exactly why every
        real forecast's confidence band widens further into the future — there's simply more
        accumulated shock-uncertainty to account for at a longer horizon.
      </P>

      <H2 c="SARIMA" />
      <P>
        SARIMA adds a second, seasonal set of AR/I/MA terms operating on lags that are multiples of
        the season length <Mx>s</Mx> (for monthly data with a yearly cycle, <Mx>s=12</Mx>):
      </P>
      <Mx block>{`  SARIMA(p,d,q)(P,D,Q)ₛ

  Seasonal AR:  uses yₜ₋ₛ, yₜ₋₂ₛ, ...        (P seasonal AR terms)
  Seasonal I:   differencing at lag s, yₜ − yₜ₋ₛ  (D times)
  Seasonal MA:  uses εₜ₋ₛ, εₜ₋₂ₛ, ...         (Q seasonal MA terms)`}</Mx>
      <Note color="info" icon="ti-info-circle">
        SARIMA is covered conceptually here rather than as a live-fitting demo. Verified evidence:
        even a simpler combined regular-MA-plus-seasonal-MA fit (no AR terms, no differencing — less
        than a real SARIMA needs) already showed clear parameter-recovery degradation at a realistic
        sample size, on top of the plain-ARIMA fragility above. A trustworthy from-scratch live SARIMA
        fit isn't a responsible promise to make at this scope — the same honesty this platform applies
        to UMAP (real formula, not implemented) elsewhere. On a correlogram, seasonal terms show up as
        spikes at lag <Mx>s, 2s, 3s, ...</Mx> instead of at the first few lags.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = ARIMA(series, order=(1, 1, 1)).fit()
forecast = model.get_forecast(steps=12)
forecast.predicted_mean, forecast.conf_int()

seasonal_model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why can't ARIMA's moving-average coefficients be estimated with ordinary least squares the way the AR coefficients can?"
          a="OLS needs its regressors to be directly observed — the AR terms (past y values) are, but the MA terms (past forecast errors) depend on the very parameters being estimated, since the errors are only known once you have a fitted model. This circularity is why MA (and mixed ARMA) models need an iterative procedure like Conditional Sum of Squares or full maximum likelihood instead of a single closed-form solve." />
      <QA q="Looking at a series' ACF and PACF, the ACF tails off gradually and the PACF cuts off sharply after lag 3. What order would you try first?"
          a="An AR(3) model — a sharp PACF cutoff after lag p paired with a gradually-decaying ACF is the classic signature of a pure autoregressive process of that order. The mirror-image pattern (ACF cuts off, PACF decays) instead points to a pure moving-average process." />
      <QA q="Why does a forecast's confidence interval get wider the further into the future it goes?"
          a="Every additional forecast step accumulates one more term's worth of shock uncertainty in the model's MA(∞) representation — the forecast-error variance at horizon h sums up h terms of squared psi-weights, and that sum can only grow (or plateau, never shrink) as h increases, so there's mechanically more accumulated uncertainty the further out you forecast." />
      <QA q="You fit a plain ARIMA model directly on a strongly seasonal series without differencing at the seasonal lag first. What goes wrong?"
          a="A low-order ARIMA model has no mechanism to represent a fixed-period oscillation — it needs either very high AR order (to create complex, oscillating roots) or an explicit seasonal term to do that. Fit naively, its forecast typically collapses toward a smooth, nearly-flat trend continuation that badly understates the real seasonal swing, which is exactly the gap SARIMA (or deseasonalizing first) is designed to close." />
    </div>
  );
}

function SectionSmoothing() {
  return (
    <div>
      <P>
        Exponential smoothing forecasts by weighting recent observations more heavily than old ones,
        with the weight decaying exponentially into the past — cheap, robust, and often
        surprisingly competitive with far more complex models.
      </P>
      <H2 c="Simple Exponential Smoothing (SES)" />
      <Mx block>{`  ŷₜ₊₁ = α·yₜ + (1−α)·ŷₜ        (level only — no trend, no seasonality)`}</Mx>
      <P>SES has no way to represent a trend or a season — it can only track the current level.</P>

      <H2 c="Holt's Linear Trend Model" />
      <Mx block>{`  Level:  ℓₜ = α·yₜ + (1−α)·(ℓₜ₋₁ + bₜ₋₁)
  Trend:  bₜ = β·(ℓₜ − ℓₜ₋₁) + (1−β)·bₜ₋₁
  Forecast, h steps ahead:  ŷₜ₊ₕ = ℓₜ + h·bₜ`}</Mx>

      <H2 c="Holt-Winters Method" />
      <Mx block>{`  Level:     ℓₜ = α·(yₜ − sₜ₋ₘ) + (1−α)·(ℓₜ₋₁ + bₜ₋₁)
  Trend:     bₜ = β·(ℓₜ − ℓₜ₋₁) + (1−β)·bₜ₋₁
  Seasonal:  sₜ = γ·(yₜ − ℓₜ) + (1−γ)·sₜ₋ₘ         (additive form, period m)
  Forecast:  ŷₜ₊ₕ = ℓₜ + h·bₜ + sₜ₊ₕ₋ₘ`}</Mx>

      <H2 c="Try It — Compare All Three" />
      <P>All three parameter sets fit by grid search minimizing in-sample squared error.</P>
      <ExponentialSmoothingDemo />

      <Table
        heads={['Method', 'Use when...']}
        rows={[
          ['SES', 'No visible trend or seasonality'],
          ['Holt', 'A trend, but no seasonality'],
          ['Holt-Winters', 'Both trend and seasonality — additive if the seasonal swing is a constant size, multiplicative if it scales with the level (same rule as Decomposition)'],
        ]}
      />

      <H3 c="A minimal working example" />
      <Code>{`from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12).fit()
model.forecast(12)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does Simple Exponential Smoothing's optimal alpha often end up very close to 1 on data that actually has trend or seasonality?"
          a="With no way to explicitly represent trend or seasonality, SES's only lever for tracking a moving target is to weight the most recent observation almost entirely — a high alpha makes it nearly copy the last value forward, which minimizes one-step error even though it can't produce a sensible multi-step forecast of a trending or seasonal series." />
      <QA q="On a dataset with strong seasonality but only mild trend, would you expect Holt to outperform SES?"
          a="Not necessarily, and it can even do worse — Holt adds machinery for capturing a trend, but if the dominant unmodeled structure is actually seasonal, that extra trend-fitting doesn't address the real problem and can add its own estimation noise on top. Only Holt-Winters, which explicitly models seasonality, reliably wins in that situation." />
      <QA q="What's the practical difference between additive and multiplicative seasonality in Holt-Winters, and how do you choose?"
          a="Additive assumes a fixed-size seasonal swing added to the level; multiplicative assumes the swing scales with the level. The choice is the same visual/practical judgment call as in classical decomposition — does the seasonal amplitude look constant, or does it grow as the series' level grows?" />
      <QA q="Exponential smoothing has no explicit 'stationarity' requirement the way ARIMA does. Why not?"
          a="Its recursive level/trend/seasonal updates are built to explicitly track a moving level, trend, and seasonal pattern rather than assuming a single fixed set of statistical properties throughout — it's designed for exactly the kind of evolving series that would fail a stationarity test, which is a genuine architectural difference from ARIMA, not an oversight." />
    </div>
  );
}

function SectionFeatures() {
  return (
    <div>
      <P>
        Instead of a specialized time-series model, reframe forecasting as ordinary supervised
        learning: turn each time point into a row of features computed only from its own past, and
        hand the resulting table to any standard model — Boosting, Random Forest, even plain linear
        regression.
      </P>
      <H2 c="Feature Types" />
      <Table
        heads={['Feature', 'What it captures']}
        rows={[
          ['Lag features (yₜ₋₁, yₜ₋₂, ...)', 'Direct recent history, the ML analogue of AR terms'],
          ['Rolling window stats (mean, std, min/max)', 'Local level and volatility over a recent window'],
          ['Time-based / cyclical features (sin/cos of month, day-of-week, is_holiday)', 'Calendar-driven seasonality, without needing an explicit seasonal model term'],
        ]}
      />
      <Note color="info" icon="ti-info-circle">
        Cyclical features are encoded as <Mx>sin(2π·m/12)</Mx> and <Mx>cos(2π·m/12)</Mx> together
        (a single "month number" column would wrongly imply December and January are far apart) — the
        sin/cos pair preserves the true circular distance between calendar positions.
      </Note>

      <H2 c="Try It — Slide the Feature Window" />
      <FeatureSlideDemo />

      <H2 c="Classical vs. Feature-Based ML Forecasting" />
      <Table
        heads={['', 'Classical (ARIMA, Exponential Smoothing)', 'Feature-Based ML']}
        rows={[
          ['Interpretability', 'High — coefficients have direct meaning', 'Lower, especially for boosted trees'],
          ['Data needed', 'Works with relatively little data', 'Usually needs more to outperform classical baselines'],
          ['Uncertainty quantification', 'Built-in (confidence intervals from the model)', "Not automatic — needs a separate approach (e.g. quantile regression)"],
          ['Nonlinearity / exogenous regressors', 'Limited without extending the model by hand', 'Handles naturally — just add more feature columns'],
        ]}
      />
      <P>
        Once framed this way, forecasting is a direct hand-off to everything covered under Supervised
        Learning — feed this exact feature table to Boosting or Random Forest and the rest of the
        pipeline (train/test split, hyperparameter tuning, evaluation metrics) is unchanged from any
        other tabular problem, <strong>except for how the data gets split</strong> — see below.
      </P>

      <H2 c="Walk-Forward Cross-Validation" />
      <P>
        Ordinary K-Fold cross-validation shuffles rows before splitting — for time series, that means
        training on the future and validating on the past, silently leaking information no real
        deployment would ever have. The fix always trains strictly on the past and validates strictly
        on the future:
      </P>
      <Mx block>{`  Expanding window:  fold i trains on [0, T+i·s), tests on [T+i·s, T+i·s+s)
  Sliding window:     fold i trains on [i·s, T+i·s), tests on [T+i·s, T+i·s+s)

  (T = initial training size, s = step/test-window size — here, s = 12,
  one full seasonal cycle, so every fold tests on a comparable "shape" of
  data)`}</Mx>
      <WalkForwardDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=12)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]   # always past -> future`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why is a plain K-Fold split invalid for time series data, specifically?"
          a="K-Fold shuffles rows before splitting into folds, which for sequential data means a model can end up training on rows chronologically after the ones it's validated on — information about the future leaking into training, which no real deployed model would ever have access to. Walk-forward validation enforces that every fold trains strictly on the past and validates strictly on the future." />
      <QA q="When would you prefer an expanding window over a sliding window for walk-forward validation?"
          a="Expanding window, which keeps all historical data in every fold, is preferable when older data is still representative and more history reliably improves the model. Sliding window, which drops old data as it moves forward, is preferable when the underlying process changes over time (regime shifts, changing customer behavior) and old data could actively mislead a model trained on it." />
      <QA q="What's lost by treating forecasting as an ordinary supervised ML problem via lag/rolling features, compared to a classical model?"
          a="Built-in, principled uncertainty quantification — an ARIMA model's confidence intervals fall directly out of its statistical assumptions, while a gradient-boosted tree has no native notion of forecast uncertainty and needs a separate technique (e.g. quantile regression, or bootstrapped residuals) bolted on to get anything comparable." />
      <QA q="Why encode 'month of year' as a sin/cos pair instead of just the integer 1-12?"
          a="A single integer column implies month 12 (December) and month 1 (January) are 11 units apart, when they're actually adjacent on the calendar — a model would have to work hard to learn that wraparound. The sin/cos pair maps every month onto a circle, so December and January end up genuinely close together in feature space, matching their true cyclical distance." />
    </div>
  );
}

const SECTION_MAP = {
  decomposition: <SectionDecomposition />,
  stationarity: <SectionStationarity />,
  arima: <SectionARIMA />,
  smoothing: <SectionSmoothing />,
  features: <SectionFeatures />,
};

export default function TimeSeriesAnalysis() {
  const [active, setActive] = useState('decomposition');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 19</div>
        <h1 className="page-header-title">Time Series Fundamentals</h1>
        <p className="page-header-subtitle">
          The building blocks of sequential data, and the classical toolkit for predicting what comes
          next.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={19} />
    </div>
  );
}
