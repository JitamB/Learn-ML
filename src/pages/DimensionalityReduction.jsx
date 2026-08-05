import { useState, useMemo, useEffect } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

/* ── Shared helpers ─────────────────────────────────────────── */
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
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
const CLUSTER_TEXT = ['var(--color-text-info)', 'var(--color-text-danger)', 'var(--color-text-success)', 'var(--color-text-warning)'];

/* ── PCA demo: real 2x2 covariance eigen-decomposition, angle slider ── */
const PCA_POINTS = (() => {
  const randA = seededRandom(17), randB = seededRandom(53);
  const pts = [];
  for (let i = 0; i < 70; i++) {
    const u = randNormalish(randA, 0, 1), v = randNormalish(randB, 0, 1);
    pts.push({ x: 5 + 2 * u, y: 5 + 1.4 * u + 1.0 * v });
  }
  return pts;
})();
const PCA_COV = (() => {
  const mx = PCA_POINTS.reduce((s, p) => s + p.x, 0) / PCA_POINTS.length;
  const my = PCA_POINTS.reduce((s, p) => s + p.y, 0) / PCA_POINTS.length;
  let sxx = 0, syy = 0, sxy = 0;
  PCA_POINTS.forEach(p => { const dx = p.x - mx, dy = p.y - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; });
  const n = PCA_POINTS.length;
  return { mx, my, sxx: sxx / n, syy: syy / n, sxy: sxy / n };
})();
const PCA_TOTAL_VAR = PCA_COV.sxx + PCA_COV.syy;
const PCA_TRUE_ANGLE = (() => {
  const angle = 0.5 * Math.atan2(2 * PCA_COV.sxy, PCA_COV.sxx - PCA_COV.syy) * 180 / Math.PI;
  return ((angle % 180) + 180) % 180;
})();
function varianceAtAngle(deg) {
  const rad = deg * Math.PI / 180;
  const ux = Math.cos(rad), uy = Math.sin(rad);
  return ux * ux * PCA_COV.sxx + 2 * ux * uy * PCA_COV.sxy + uy * uy * PCA_COV.syy;
}
const PCA_XS = PCA_POINTS.map(p => p.x), PCA_YS = PCA_POINTS.map(p => p.y);
const PCA_X_MIN = Math.min(...PCA_XS) - 1, PCA_X_MAX = Math.max(...PCA_XS) + 1;
const PCA_Y_MIN = Math.min(...PCA_YS) - 1, PCA_Y_MAX = Math.max(...PCA_YS) + 1;
const PCA_SVG_W = 400, PCA_SVG_H = 280;
const pcaXToPx = x => (x - PCA_X_MIN) / (PCA_X_MAX - PCA_X_MIN) * PCA_SVG_W;
const pcaYToPx = y => PCA_SVG_H - (y - PCA_Y_MIN) / (PCA_Y_MAX - PCA_Y_MIN) * PCA_SVG_H;

function PCADemo() {
  const [angle, setAngle] = useState(10);
  const rad = angle * Math.PI / 180;
  const ux = Math.cos(rad), uy = Math.sin(rad);
  const varThis = varianceAtAngle(angle);
  const varPerp = PCA_TOTAL_VAR - varThis;
  const pctThis = varThis / PCA_TOTAL_VAR * 100;
  const pctPerp = 100 - pctThis;
  const L = Math.max(PCA_X_MAX - PCA_X_MIN, PCA_Y_MAX - PCA_Y_MIN);

  const lineX1 = PCA_COV.mx - ux * L, lineY1 = PCA_COV.my - uy * L;
  const lineX2 = PCA_COV.mx + ux * L, lineY2 = PCA_COV.my + uy * L;

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
        <button onClick={() => setAngle(Math.round(PCA_TRUE_ANGLE))} style={toggleBtnStyle(Math.abs(angle - PCA_TRUE_ANGLE) < 1)}>Snap to PC1</button>
      </div>
      <SliderRow label="Direction angle" min={0} max={180} step={1} value={angle} onChange={setAngle} fmt={v => `${v}°`} />
      <Grid cols={2} gap={12}>
        <div>
          <svg viewBox={`0 0 ${PCA_SVG_W} ${PCA_SVG_H}`} style={{ width: '100%', height: 'auto', display: 'block', border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)' }}>
            <line x1={pcaXToPx(lineX1)} y1={pcaYToPx(lineY1)} x2={pcaXToPx(lineX2)} y2={pcaYToPx(lineY2)} stroke="var(--color-text-danger)" strokeWidth={1.5} />
            {PCA_POINTS.map((p, i) => {
              const proj = (p.x - PCA_COV.mx) * ux + (p.y - PCA_COV.my) * uy;
              const projX = PCA_COV.mx + proj * ux, projY = PCA_COV.my + proj * uy;
              return (
                <g key={i}>
                  <line x1={pcaXToPx(p.x)} y1={pcaYToPx(p.y)} x2={pcaXToPx(projX)} y2={pcaYToPx(projY)} stroke="var(--color-border-tertiary)" strokeWidth={0.75} opacity={0.6} />
                  <circle cx={pcaXToPx(projX)} cy={pcaYToPx(projY)} r={1.6} fill="var(--color-text-danger)" />
                  <circle cx={pcaXToPx(p.x)} cy={pcaYToPx(p.y)} r={2.6} fill="var(--color-border-info)" />
                </g>
              );
            })}
          </svg>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 14 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>This direction ({angle}°)</div>
            <div style={{ height: 20, borderRadius: 'var(--border-radius-sm)', background: 'var(--color-background-secondary)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${pctThis}%`, background: 'var(--color-background-info)', borderRight: '2px solid var(--color-border-info)' }} />
            </div>
            <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', marginTop: 2 }}>{pctThis.toFixed(1)}% of total variance</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>Perpendicular direction ({((angle + 90) % 180).toFixed(0)}°)</div>
            <div style={{ height: 20, borderRadius: 'var(--border-radius-sm)', background: 'var(--color-background-secondary)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${pctPerp}%`, background: 'var(--color-background-danger)', borderRight: '2px solid var(--color-border-danger)' }} />
            </div>
            <div style={{ fontSize: 12, fontFamily: 'var(--font-mono)', marginTop: 2 }}>{pctPerp.toFixed(1)}% of total variance</div>
          </div>
        </div>
      </Grid>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Blue dots = actual points; red dots = their shadow ("projection") onto the red line. The two
        bars always sum to 100% — that's a real mathematical identity, true at every angle, not just
        at the optimum. PCA is simply the angle where that split is most lopsided.{' '}
        {Math.abs(angle - PCA_TRUE_ANGLE) < 1 ? "You're at PC1 — the single most lopsided angle there is." : ''}
      </div>
    </VizBox>
  );
}

/* ── t-SNE demo: real gradient descent (Van der Maaten & Hinton, 2008) ── */
const TSNE_DIMS = 5, TSNE_CLUSTERS = 4, TSNE_PER_CLUSTER = 11, TSNE_N = TSNE_CLUSTERS * TSNE_PER_CLUSTER;
const TSNE_MAX_ITER = 400;
const TSNE_POINTS = (() => {
  const rands = [901, 902, 903, 904, 905].map(seed => seededRandom(seed));
  const pts = [];
  for (let c = 0; c < TSNE_CLUSTERS; c++) {
    const center = Array.from({ length: TSNE_DIMS }, (_, d) => (d === c ? 10 : 0));
    for (let i = 0; i < TSNE_PER_CLUSTER; i++) {
      pts.push({ vec: center.map((cv, d) => randNormalish(rands[d], cv, 1.5)), cluster: c });
    }
  }
  return pts;
})();
const TSNE_D2 = TSNE_POINTS.map(p => TSNE_POINTS.map(q => {
  let s = 0; for (let d = 0; d < TSNE_DIMS; d++) { const diff = p.vec[d] - q.vec[d]; s += diff * diff; } return s;
}));
function computeTSNE_P(perplexity) {
  const targetEntropy = Math.log(perplexity);
  const N = TSNE_N;
  const P = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) {
    let lo = 1e-5, hi = 1e5, beta = 1.0;
    let condP = new Array(N).fill(0);
    for (let iter = 0; iter < 60; iter++) {
      let sumP = 0;
      for (let j = 0; j < N; j++) {
        if (j === i) { condP[j] = 0; continue; }
        condP[j] = Math.exp(-TSNE_D2[i][j] * beta);
        sumP += condP[j];
      }
      let H = 0;
      for (let j = 0; j < N; j++) { if (j === i || condP[j] <= 1e-12) continue; const p = condP[j] / sumP; H -= p * Math.log(p); }
      const diff = H - targetEntropy;
      if (Math.abs(diff) < 1e-6) break;
      if (diff > 0) lo = beta; else hi = beta;
      beta = (lo + hi) / 2;
    }
    let sumP = 0; for (let j = 0; j < N; j++) sumP += condP[j];
    for (let j = 0; j < N; j++) P[i][j] = condP[j] / sumP;
  }
  const Psym = Array.from({ length: N }, () => new Array(N).fill(0));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) Psym[i][j] = (P[i][j] + P[j][i]) / (2 * N);
  return Psym;
}
function makeInitialTSNEEmbedding(seed) {
  const rand = seededRandom(seed);
  return {
    Y: Array.from({ length: TSNE_N }, () => [randNormalish(rand, 0, 1) * 0.01, randNormalish(rand, 0, 1) * 0.01]),
    vel: Array.from({ length: TSNE_N }, () => [0, 0]),
    iteration: 0,
  };
}
function tsneAdvance(P, embedding, numSteps) {
  let { Y, vel, iteration } = embedding;
  Y = Y.map(p => [...p]);
  vel = vel.map(v => [...v]);
  const N = TSNE_N, lr = 50;
  for (let s = 0; s < numSteps && iteration < TSNE_MAX_ITER; s++) {
    const exaggeration = iteration < 100 ? 4 : 1;
    const momentum = iteration < 250 ? 0.5 : 0.8;
    const numer = Array.from({ length: N }, () => new Array(N).fill(0));
    let Z = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const dx = Y[i][0] - Y[j][0], dy = Y[i][1] - Y[j][1];
        const v = 1 / (1 + dx * dx + dy * dy);
        numer[i][j] = v; Z += v;
      }
    }
    const grad = Array.from({ length: N }, () => [0, 0]);
    for (let i = 0; i < N; i++) {
      let gx = 0, gy = 0;
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const q = numer[i][j] / Z;
        const mult = 4 * (exaggeration * P[i][j] - q) * numer[i][j];
        gx += mult * (Y[i][0] - Y[j][0]);
        gy += mult * (Y[i][1] - Y[j][1]);
      }
      grad[i] = [gx, gy];
    }
    for (let i = 0; i < N; i++) {
      vel[i][0] = momentum * vel[i][0] - lr * grad[i][0];
      vel[i][1] = momentum * vel[i][1] - lr * grad[i][1];
      Y[i][0] += vel[i][0]; Y[i][1] += vel[i][1];
    }
    const mx = Y.reduce((s, p) => s + p[0], 0) / N, my = Y.reduce((s, p) => s + p[1], 0) / N;
    Y = Y.map(p => [p[0] - mx, p[1] - my]);
    iteration++;
  }
  return { Y, vel, iteration };
}
function tsneSeparationRatio(Y) {
  let withinSum = 0, withinN = 0, betweenSum = 0, betweenN = 0;
  for (let i = 0; i < TSNE_N; i++) {
    for (let j = i + 1; j < TSNE_N; j++) {
      const d = Math.hypot(Y[i][0] - Y[j][0], Y[i][1] - Y[j][1]);
      if (TSNE_POINTS[i].cluster === TSNE_POINTS[j].cluster) { withinSum += d; withinN++; } else { betweenSum += d; betweenN++; }
    }
  }
  return withinN && betweenN ? (betweenSum / betweenN) / (withinSum / withinN) : 1;
}

function TSNEDemo() {
  const [perplexity, setPerplexity] = useState(10);
  const [resetTick, setResetTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [embedding, setEmbedding] = useState(() => makeInitialTSNEEmbedding(2000));
  const P = useMemo(() => computeTSNE_P(perplexity), [perplexity]);

  // Re-initialize whenever perplexity changes (its affinities no longer match the running
  // embedding) or Reset is clicked. A proper effect rather than a render-phase state
  // adjustment — the latter interacts badly with the Play interval under StrictMode's
  // double-invocation (a stale interval tick can land between the two renders and clobber
  // the freshly-reset embedding right back to its pre-reset value).
  useEffect(() => {
    const embeddingSeed = 2000 + Math.round(perplexity * 97) + resetTick * 13;
    setEmbedding(makeInitialTSNEEmbedding(embeddingSeed));
    setPlaying(false);
  }, [perplexity, resetTick]);

  useEffect(() => {
    if (!playing) return undefined;
    const id = setInterval(() => {
      setEmbedding(e => {
        if (e.iteration >= TSNE_MAX_ITER) { setPlaying(false); return e; }
        return tsneAdvance(P, e, 5);
      });
    }, 60);
    return () => clearInterval(id);
  }, [playing, P]);

  const ratio = useMemo(() => tsneSeparationRatio(embedding.Y), [embedding.Y]);
  const xs = embedding.Y.map(p => p[0]), ys = embedding.Y.map(p => p[1]);
  const xMin = Math.min(...xs) - 5, xMax = Math.max(...xs) + 5;
  const yMin = Math.min(...ys) - 5, yMax = Math.max(...ys) + 5;
  const W = 380, H = 280;
  const toPx = (x, y) => [(x - xMin) / (xMax - xMin) * W, H - (y - yMin) / (yMax - yMin) * H];

  return (
    <VizBox>
      <SliderRow label="Perplexity" min={2} max={20} step={1} value={perplexity} onChange={setPerplexity} fmt={v => `${v}`} />
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '10px 0' }}>
        <button onClick={() => setPlaying(p => !p)} style={toggleBtnStyle(playing)}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => setEmbedding(e => tsneAdvance(P, e, 5))} disabled={playing || embedding.iteration >= TSNE_MAX_ITER} style={{ ...toggleBtnStyle(false), opacity: (playing || embedding.iteration >= TSNE_MAX_ITER) ? 0.4 : 1 }}>Step</button>
        <button onClick={() => setResetTick(t => t + 1)} style={toggleBtnStyle(false)}>Reset</button>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--color-text-tertiary)', marginLeft: 'auto' }}>
          Iteration {embedding.iteration} / {TSNE_MAX_ITER}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block', border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', background: 'var(--color-background-secondary)' }}>
        {embedding.Y.map((p, i) => {
          const [px, py] = toPx(p[0], p[1]);
          return <circle key={i} cx={px} cy={py} r={4} fill={CLUSTER_TEXT[TSNE_POINTS[i].cluster]} />;
        })}
      </svg>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        44 points living in 5 dimensions (4 true clusters, invisible to us directly) — colors show
        the true cluster, unknown to the algorithm itself. Cluster separation ratio: {ratio.toFixed(1)}×
        (between-cluster distance ÷ within-cluster distance — started near 1× at random init).{' '}
        {embedding.iteration >= TSNE_MAX_ITER ? 'Converged.' : playing ? 'Unfolding…' : 'Press Play to watch it unfold.'}
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'pca', label: 'PCA', sub: 'Linear variance-maximizing projection' },
  { id: 'tsne', label: 't-SNE & UMAP', sub: 'Non-linear manifold learning' },
];

function SectionPCA() {
  return (
    <div>
      <P>
        Imagine shining a flashlight at a 3D object and looking at its shadow on the wall — rotate
        the object and the shadow's shape changes, even though the object itself hasn't. Principal
        Component Analysis finds the rotation that casts the <em>most informative</em> possible
        shadow: the one that spreads the data out as much as possible, so the fewest dimensions
        lose the least information.
      </P>

      <H2 c="Eigenvectors and Eigenvalues" />
      <P>
        Formally, PCA finds the eigenvectors of the data's covariance matrix. Each eigenvector is a
        direction in feature space; its corresponding eigenvalue is exactly how much variance the
        data has along that direction. Sorting eigenvectors by eigenvalue, largest first, gives an
        ordered list of "most informative direction," "second most informative direction," and so
        on — the Principal Components.
      </P>
      <Note color="info" icon="ti-info-circle">
        For 2 features, the covariance matrix's eigen-decomposition has a clean closed form —
        no iterative solver needed: eigenvalues = <Mx>(a+c)/2 ± √(((a-c)/2)² + b²)</Mx> for a
        covariance matrix <Mx>[[a,b],[b,c]]</Mx>. The interactive demo below computes this exactly,
        live, on 70 real points.
      </Note>

      <H2 c="Use Cases" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="Compression">Keep only the top few components and discard the rest — most of the information survives in far fewer numbers.</Card>
        <Card color="success" title="Noise filtering">Low-variance components are often mostly noise; dropping them can clean up a signal.</Card>
        <Card color="purple" title="Decorrelation">Principal components are, by construction, uncorrelated with each other — useful before models (like linear regression) that assume independent features.</Card>
      </Grid>

      <H2 c="Try It — Find the Principal Component by Hand" />
      <P>70 correlated points. Sweep the angle and watch the two bars — PCA is nothing more than the angle where that split is most unequal. Click "Snap to PC1" to check yourself against the real eigen-decomposition.</P>
      <PCADemo />
      <Note color="warning" icon="ti-alert-triangle">
        <strong>PCA is scale-sensitive.</strong> Because it maximizes variance, a feature measured
        in units that happen to produce larger numbers (income in dollars vs. age in years) will
        dominate the covariance matrix regardless of its actual importance — always standardize
        features to zero mean and unit variance before applying PCA, the same lesson as KNN's
        distance calculations.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)`}</Code>

      <Note color="success" icon="ti-arrow-right">
        How much variance to keep — and the explained variance ratio used to decide it — is
        covered in "Metrics" under Model Evaluation & Validation.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why must features be standardized before PCA, but not necessarily before a decision tree?"
          a="PCA finds directions of maximum variance, and variance is entirely dependent on a feature's numeric scale — a feature measured in the thousands will mechanically dominate the covariance matrix over one measured in single digits, regardless of actual signal strength. A decision tree instead compares one feature to a threshold at a time, a comparison that's invariant to the feature's overall scale, so it doesn't have this sensitivity." />
      <QA q="What does it mean for two principal components to be uncorrelated, and why does PCA guarantee this?"
          a="Uncorrelated means knowing a point's position along one principal component gives no linear information about its position along another. This falls directly out of the eigen-decomposition: eigenvectors of a symmetric matrix (which any covariance matrix is) are mathematically guaranteed to be orthogonal to each other, and projecting onto orthogonal directions produces uncorrelated coordinates." />
      <QA q="If the first 2 principal components explain 95% of the variance in a 50-feature dataset, what does that actually tell you — and not tell you?"
          a="It tells you the data's true underlying structure is far lower-dimensional than the raw feature count suggests — most of the spread in the data lives along just 2 directions. It does NOT tell you those 2 components are individually meaningful or interpretable (a component is usually some abstract mixture of many original features), and it doesn't guarantee those 2 dimensions are the most useful ones for a specific downstream task like classification, since PCA optimizes for variance, not for any particular label." />
      <QA q="Why is PCA considered a linear technique, and what's the practical consequence of that?"
          a="Every principal component is a fixed linear combination (a weighted sum) of the original features — there's no way to express a curved or twisted relationship using linear combinations alone. The practical consequence is that PCA can badly compress data that lies on a curved manifold (imagine a Swiss-roll shape) into a much less useful flat projection, which is exactly the gap that non-linear techniques like t-SNE and UMAP are built to fill." />
    </div>
  );
}

function SectionTSNE() {
  return (
    <div>
      <P>
        PCA is honest about being linear — it can only ever rotate and flatten. Real high-dimensional
        data often lies on a curved, twisted "manifold" that no flat projection represents well.
        t-SNE and UMAP were both built specifically to visualize that kind of structure in 2D or 3D.
      </P>

      <H2 c="t-SNE (t-Distributed Stochastic Neighbor Embedding)" />
      <P>
        t-SNE's core idea: convert distances into <em>probabilities</em> of being neighbors, once in
        the original high-dimensional space and once in the low-dimensional layout, then move points
        in the layout until those two sets of probabilities match as closely as possible.
      </P>
      <Mx block>{`  1. In high-D: for each point, convert distances to nearby points into
     a probability distribution (closer = more likely to be picked as
     a "neighbor"). A parameter called perplexity controls the
     effective number of neighbors considered.
  2. In low-D (2D): start from a random layout, define an analogous
     neighbor-probability distribution there too.
  3. Move the low-D points via gradient descent to make the two
     distributions match as closely as possible — points that were
     close in high-D get pulled together, far points pushed apart.`}</Mx>
      <Note color="info" icon="ti-info-circle">
        The demo below runs this exact algorithm for real (including the two standard tricks that
        make it work reliably: an early "exaggeration" phase that encourages tight initial clusters,
        and momentum that speeds up consistent movement) — not a pre-baked animation.
      </Note>

      <H2 c="Try It — Watch the Embedding Unfold" />
      <P>44 points that live in 5 dimensions across 4 true clusters — far too many dimensions to plot directly. Press Play and watch random noise organize into visible groups.</P>
      <TSNEDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)   # for visualization only — not for feeding into another model`}</Code>

      <H2 c="UMAP (Uniform Manifold Approximation and Projection)" />
      <P>
        UMAP tackles the same problem with different machinery — a mathematical framework based on
        "fuzzy simplicial sets" rather than probability distributions — and has become the more
        common default in recent years.
      </P>
      <Table
        heads={['', 't-SNE', 'UMAP']}
        rows={[
          ['Key parameter', 'Perplexity (effective neighbor count)', 'n_neighbors (similar role) + min_dist'],
          ['Speed', 'Slower — scales poorly past tens of thousands of points', 'Substantially faster, scales to much larger datasets'],
          ['Structure preserved', 'Prioritizes local structure; global distances between clusters aren’t meaningful', 'Better balance of local and global structure'],
          ['Determinism', 'Different runs (different random init) can look different', 'Also stochastic, but tends to be more stable run-to-run'],
        ]}
      />
      <Note color="warning" icon="ti-alert-triangle">
        A common misreading of both t-SNE and UMAP plots: the <em>distance between clusters</em> and
        even relative cluster <em>sizes</em> in the 2D output are not reliably meaningful — only
        which points ended up near each other tends to be trustworthy. Don't read "cluster A is
        twice as far from B as from C" as a real quantitative statement.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why can't you use a t-SNE or UMAP embedding as input features for a downstream supervised model?"
          a="Both techniques are fit specifically to produce a good 2D/3D visualization of the training data they saw — there's no principled way to ‘transform’ a brand-new point into that same embedding space the way PCA's linear projection can be reapplied to new data. They're visualization and exploration tools, not general-purpose dimensionality-reduction preprocessing steps." />
      <QA q="What happens to a t-SNE plot if perplexity is set far too low or far too high?"
          a="Too low, and each point only 'sees' a handful of its absolute closest neighbors, which can shatter genuine clusters into many small fragments. Too high, and the neighbor notion becomes so broad that distinct clusters can blur into each other, since points are effectively being compared against most of the dataset at once rather than their true local neighborhood." />
      <QA q="Why does t-SNE need an 'early exaggeration' phase at the start of optimization?"
          a="Multiplying the high-dimensional probabilities by a constant factor early in training makes genuine neighbors pull together much more aggressively than they otherwise would, helping true clusters form tight, well-separated groups before the optimization settles down — without it, the layout is more likely to get stuck with clusters overlapping or interleaved." />
      <QA q="If UMAP is generally faster and preserves more global structure, why does t-SNE remain widely used?"
          a="t-SNE is older, extremely well-studied, and its failure modes are well understood and documented, which matters when explaining results to others; it also remains a very strong choice specifically for revealing fine local structure. In practice many practitioners try both and compare, rather than treating either as a strict replacement for the other." />
    </div>
  );
}

const SECTION_MAP = {
  pca: <SectionPCA />,
  tsne: <SectionTSNE />,
};

export default function DimensionalityReduction() {
  const [active, setActive] = useState('pca');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 10</div>
        <h1 className="page-header-title">Dimensionality Reduction</h1>
        <p className="page-header-subtitle">
          Compress high-dimensional data into fewer dimensions, for visualization, noise reduction,
          or as a preprocessing step for other models.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={10} />
    </div>
  );
}
