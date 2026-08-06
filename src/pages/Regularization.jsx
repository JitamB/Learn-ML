import { useState, useMemo } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
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

/* ── Small hand-solved linear algebra — same precedent as           ──
   DimensionalityReduction's closed-form covariance work, generalized
   to n x n via Gauss-Jordan for Ridge's normal equation. */
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
function softThreshold(z, gamma) { return Math.sign(z) * Math.max(Math.abs(z) - gamma, 0); }

/* ── Regularization demo dataset ────────────────────────────────
   n=10, p=5: v1,v2 genuinely predictive of y; v3 = v1 + small noise
   (redundant/correlated with v1, verified corr(x1,x3)=0.988); v4,v5
   pure noise. Standardized to mean 0 / std 1 per column, y centered. */
const REG_RAW = (() => {
  const rand = seededRandom(909);
  return Array.from({ length: 10 }, () => {
    const v1 = randNormalish(rand, 0, 2), v2 = randNormalish(rand, 0, 2);
    const v3 = v1 + randNormalish(rand, 0, 0.4);
    const v4 = randNormalish(rand, 0, 2), v5 = randNormalish(rand, 0, 2);
    const y = 3 * v1 + 1.5 * v2 + randNormalish(rand, 0, 1);
    return { x: [v1, v2, v3, v4, v5], y };
  });
})();
const { REG_X, REG_Y } = (() => {
  const n = REG_RAW.length, p = REG_RAW[0].x.length;
  const means = Array.from({ length: p }, (_, j) => REG_RAW.reduce((s, r) => s + r.x[j], 0) / n);
  const stds = Array.from({ length: p }, (_, j) => Math.sqrt(REG_RAW.reduce((s, r) => s + (r.x[j] - means[j]) ** 2, 0) / n));
  const yMean = REG_RAW.reduce((s, r) => s + r.y, 0) / n;
  return {
    REG_X: REG_RAW.map(r => r.x.map((v, j) => (v - means[j]) / stds[j])),
    REG_Y: REG_RAW.map(r => r.y - yMean),
  };
})();
const REG_P = 5, REG_N = REG_X.length;
const FEATURE_LABELS = ['x1 (predictive)', 'x2 (predictive)', 'x3 (redundant)', 'x4 (noise)', 'x5 (noise)'];
const FEATURE_COLORS = ['var(--color-background-info)', 'var(--color-background-info)', 'var(--color-background-warning)', 'var(--color-background-tertiary)', 'var(--color-background-tertiary)'];

function ridgeFit(lambda) {
  const XtX = Array.from({ length: REG_P }, (_, i) =>
    Array.from({ length: REG_P }, (_, j) => REG_X.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? lambda : 0)));
  const Xty = Array.from({ length: REG_P }, (_, i) => REG_X.reduce((s, row, k) => s + row[i] * REG_Y[k], 0));
  return solveLinearSystem(XtX, Xty);
}
const REG_COL_NORM_SQ = Array.from({ length: REG_P }, (_, j) => REG_X.reduce((s, row) => s + row[j] * row[j], 0));
function coordDescentFit(lambda, alpha, iters = 1500) {
  let beta = new Array(REG_P).fill(0);
  for (let it = 0; it < iters; it++) {
    for (let j = 0; j < REG_P; j++) {
      let rho = 0;
      for (let i = 0; i < REG_N; i++) {
        let pred = 0;
        for (let k = 0; k < REG_P; k++) if (k !== j) pred += REG_X[i][k] * beta[k];
        rho += REG_X[i][j] * (REG_Y[i] - pred);
      }
      beta[j] = softThreshold(rho, lambda * alpha * REG_N) / (REG_COL_NORM_SQ[j] + lambda * (1 - alpha) * REG_N);
    }
  }
  return beta;
}

const MODE_RANGES = {
  ridge: { min: 0, max: 60, step: 0.5, default: 2 },
  lasso: { min: 0, max: 7, step: 0.05, default: 0.3 },
  elastic: { min: 0, max: 16, step: 0.1, default: 4 },
};

function RegularizationDemo() {
  const [mode, setMode] = useState('ridge');
  const [lambda, setLambda] = useState(MODE_RANGES.ridge.default);
  const [alpha, setAlpha] = useState(0.5);

  const beta = useMemo(() => {
    if (mode === 'ridge') return ridgeFit(lambda);
    if (mode === 'lasso') return coordDescentFit(lambda, 1);
    return coordDescentFit(lambda, alpha);
  }, [mode, lambda, alpha]);

  function switchMode(m) {
    setMode(m);
    setLambda(MODE_RANGES[m].default);
  }

  const maxAbs = 8;
  const barH = 90;

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap' }}>
        <button onClick={() => switchMode('ridge')} style={toggleBtnStyle(mode === 'ridge')}>Ridge (L2)</button>
        <button onClick={() => switchMode('lasso')} style={toggleBtnStyle(mode === 'lasso')}>Lasso (L1)</button>
        <button onClick={() => switchMode('elastic')} style={toggleBtnStyle(mode === 'elastic')}>Elastic Net</button>
      </div>
      <SliderRow label="λ (penalty strength)" min={MODE_RANGES[mode].min} max={MODE_RANGES[mode].max} step={MODE_RANGES[mode].step} value={lambda} onChange={setLambda} fmt={v => v.toFixed(2)} />
      {mode === 'elastic' && (
        <SliderRow label="α (1 = pure Lasso, 0 = pure Ridge)" min={0.1} max={0.9} step={0.1} value={alpha} onChange={setAlpha} fmt={v => v.toFixed(1)} />
      )}
      <div style={{ display: 'flex', gap: 18, alignItems: 'flex-end', height: barH + 40, marginTop: 16, paddingTop: 20, borderTop: '1px solid var(--color-border-tertiary)' }}>
        {beta.map((b, j) => {
          const h = Math.min(barH, Math.abs(b) / maxAbs * barH);
          return (
            <div key={j} style={{ textAlign: 'center', width: 56 }}>
              <div style={{ height: barH, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', position: 'relative' }}>
                <div style={{
                  height: h, width: 40, margin: '0 auto', background: FEATURE_COLORS[j],
                  border: '1px solid var(--color-border-tertiary)', borderRadius: '3px 3px 0 0',
                  opacity: Math.abs(b) < 0.005 ? 0.25 : 1,
                }} />
              </div>
              <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', marginTop: 4 }}>{b.toFixed(2)}</div>
              <div style={{ fontSize: 10, color: 'var(--color-text-tertiary)' }}>{FEATURE_LABELS[j]}</div>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 14 }}>
        {mode === 'ridge' && 'Ridge shrinks every coefficient smoothly toward zero as λ grows, but never sets one exactly to zero — even at λ=60, x4 and x5 are small but nonzero. Try λ=0: x3\'s coefficient comes out wrong-signed, a classic multicollinearity symptom (x1 and x3 are correlated 0.99) — any real penalty corrects it.'}
        {mode === 'lasso' && 'Lasso kills x3, x4, and x5 exactly at λ≈0.3, then only x1 survives past λ≈3.5, then everything hits exactly zero by λ=7 — a genuine built-in feature selector.'}
        {mode === 'elastic' && 'At α=0.5, x3 (correlated with x1) stays alive alongside x1 across a wide λ range instead of being arbitrarily zeroed like pure Lasso does — Elastic Net\'s "grouping effect": correlated predictors are kept or dropped together. Push α toward 0.9 and watch x2 fade out while the correlated x1/x3 pair keeps surviving together.'}
      </div>
    </VizBox>
  );
}

/* ── Geometric interpretation: L1 diamond vs L2 circle ──────────
   A standard illustrative textbook diagram (ESL-style) — pure geometry,
   not a fitted fitted result: elliptical loss contours around an
   unconstrained optimum outside the constraint region, with the L1
   solution placed exactly at a diamond corner (one coefficient = 0)
   and the L2 solution at a generic smooth point on the circle. */
function GeometryDemo() {
  const [mode, setMode] = useState('l1');
  const cx = 140, cy = 140;
  const trueOpt = { x: cx + 95, y: cy - 65 };
  const t = 78;

  const diamond = `${cx + t},${cy} ${cx},${cy - t} ${cx - t},${cy} ${cx},${cy + t}`;
  const l1Solution = { x: cx + t, y: cy };
  const angle = (35 * Math.PI) / 180;
  const l2Solution = { x: cx + 65 * Math.cos(angle), y: cy - 65 * Math.sin(angle) };

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <button onClick={() => setMode('l1')} style={toggleBtnStyle(mode === 'l1')}>L1 constraint (diamond)</button>
        <button onClick={() => setMode('l2')} style={toggleBtnStyle(mode === 'l2')}>L2 constraint (circle)</button>
      </div>
      <svg viewBox="0 0 280 280" width={280} height={280} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
        <line x1={0} y1={cy} x2={280} y2={cy} stroke="var(--color-border-tertiary)" strokeWidth={1} />
        <line x1={cx} y1={0} x2={cx} y2={280} stroke="var(--color-border-tertiary)" strokeWidth={1} />
        {[30, 55, 80].map(r => (
          <ellipse key={r} cx={trueOpt.x} cy={trueOpt.y} rx={r * 1.3} ry={r * 0.8}
            fill="none" stroke="var(--color-border-danger)" strokeWidth={1} opacity={0.5} />
        ))}
        <circle cx={trueOpt.x} cy={trueOpt.y} r={3} fill="var(--color-text-danger)" />
        <text x={trueOpt.x + 8} y={trueOpt.y - 8} fontSize={10} fill="var(--color-text-danger)">unconstrained optimum</text>
        {mode === 'l1' ? (
          <polygon points={diamond} fill="var(--color-background-info)" fillOpacity={0.25} stroke="var(--color-border-info)" strokeWidth={2} />
        ) : (
          <circle cx={cx} cy={cy} r={65} fill="var(--color-background-info)" fillOpacity={0.25} stroke="var(--color-border-info)" strokeWidth={2} />
        )}
        {mode === 'l1' ? (
          <circle cx={l1Solution.x} cy={l1Solution.y} r={5} fill="var(--color-text-success)" stroke="white" strokeWidth={1.5} />
        ) : (
          <circle cx={l2Solution.x} cy={l2Solution.y} r={5} fill="var(--color-text-success)" stroke="white" strokeWidth={1.5} />
        )}
      </svg>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        {mode === 'l1'
          ? 'The diamond has corners sitting exactly on the axes — the smallest loss contour touching the diamond very often touches it right at a corner, which means one coefficient lands at exactly zero. That\'s the geometric reason L1 produces sparsity.'
          : 'The circle has no corners anywhere — the smallest touching contour lands at some generic smooth point, essentially never exactly on an axis. Both coefficients shrink, but neither is singled out to be eliminated.'}
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'penalties', label: 'L1, L2 & Elastic Net', sub: 'Penalizing complexity, and watching weights react live' },
  { id: 'theory', label: 'Why Regularization Works', sub: 'The Bayesian view, and its bias-variance trade' },
];

function SectionPenalties() {
  return (
    <div>
      <P>
        Every model in this platform so far has minimized some loss function purely on the training
        data — nothing in that objective cares whether the resulting coefficients are large, unstable,
        or dependent on noise. Regularization adds a second term to the objective that explicitly
        penalizes complexity, trading a little bit of training-set fit for a model that generalizes
        more reliably.
      </P>

      <H2 c="The Three Penalties" />
      <Mx block>{`  Ordinary loss:              minimize  Loss(β)
  L2 (Ridge):                 minimize  Loss(β) + λ Σ βⱼ²
  L1 (Lasso):                 minimize  Loss(β) + λ Σ |βⱼ|
  Elastic Net:                minimize  Loss(β) + λ [ α Σ|βⱼ| + (1−α)/2 · Σ βⱼ² ]

  λ controls overall penalty strength; α (Elastic Net only) blends
  between pure Lasso (α=1) and pure Ridge (α=0).`}</Mx>

      <H2 c="Geometric Interpretation — Why L1 Creates Zeros and L2 Doesn't" />
      <P>
        Minimizing a penalized loss is equivalent to minimizing the unpenalized loss subject to a
        hard constraint on the coefficients' size — <Mx>Σ|βⱼ| ≤ t</Mx> for L1, <Mx>Σβⱼ² ≤ t²</Mx> for
        L2. Picture the unpenalized loss's elliptical contours (bigger ellipse = worse fit) shrinking
        inward from the true unconstrained optimum until they just touch the constraint region — that
        touching point is the penalized solution.
      </P>
      <GeometryDemo />

      <H2 c="Practical Notes" />
      <Grid cols={2} gap={10}>
        <Card color="warning" title="Feature scaling is required">
          A penalty term sums coefficients directly — a feature measured in millions gets a tiny
          coefficient and barely feels the penalty, while a feature measured in single digits gets
          hammered, purely because of units, not actual importance. Always standardize features before
          applying any of these penalties.
        </Card>
        <Card color="info" title="Choosing λ">
          λ is itself a hyperparameter — picked by cross-validation, the same way any other
          hyperparameter is, covered in full under Hyperparameter Tuning.
        </Card>
      </Grid>

      <H2 c="Try It — Watch Coefficients React to Each Penalty" />
      <P>
        A real fit on 10 samples, 5 features: two features genuinely drive the target, one is a noisy
        near-duplicate of a real predictor (correlation 0.99), and two are pure noise. Ridge is solved
        closed-form (the normal equation with an added <Mx>λI</Mx> term); Lasso and Elastic Net are
        solved by real coordinate descent with soft-thresholding, iterated to convergence.
      </P>
      <RegularizationDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X_train)   # always scale first

ridge = Ridge(alpha=1.0).fit(X_scaled, y_train)         # alpha here = our λ
lasso = Lasso(alpha=0.1).fit(X_scaled, y_train)         # smaller λ — Lasso is more aggressive per unit
enet  = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_scaled, y_train)  # l1_ratio = our α`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Ridge has a closed-form solution, but Lasso needs iterative coordinate descent. Why the difference, given both are convex?"
          a="L2's penalty (Σβⱼ²) is smooth and differentiable everywhere, so setting the gradient of the full penalized loss to zero is still just linear algebra — the normal equation with an extra λI term. L1's penalty (Σ|βⱼ|) has a sharp, non-differentiable kink at zero, so there's no single gradient to set to zero at that point — the objective is still convex, but it needs a subgradient-aware iterative method like coordinate descent instead of one algebraic step." />
      <QA q="Why does Lasso tend to produce sparse solutions (some coefficients exactly zero) while Ridge doesn't?"
          a="Geometrically, the L1 penalty's constraint region is a diamond with corners sitting exactly on the coordinate axes, so the loss contour touching that region very often does so right at a corner — meaning one or more coefficients land at exactly zero. The L2 penalty's constraint region is a smooth circle with no corners, so its touching point is essentially never exactly on an axis, and coefficients shrink without ever being fully eliminated." />
      <QA q="You have two features that are highly correlated with each other. How do Ridge, Lasso, and Elastic Net each handle that differently?"
          a="Lasso tends to arbitrarily pick one of the two and zero out the other, even though they're almost equally informative — which one survives can be sensitive to noise. Ridge shrinks both together, roughly evenly. Elastic Net splits the difference: its L2 component keeps correlated features grouped together (both survive, together, unless the overall penalty is very Lasso-heavy), while its L1 component still zeroes out features that are genuinely irrelevant." />
      <QA q="Why is it necessary to scale features before applying L1 or L2 regularization, but not strictly necessary for plain unregularized linear regression?"
          a="Unregularized regression's coefficients naturally absorb each feature's units — a feature in millions just gets a proportionally tiny coefficient, and the fit is unaffected. But a penalty term sums coefficients directly, so an unscaled large-units feature would be penalized far less than an unscaled small-units feature purely because of arbitrary measurement units, distorting which features the penalty actually suppresses." />
    </div>
  );
}

function SectionTheory() {
  return (
    <div>
      <P>
        Regularization can be derived two different ways that arrive at the exact same formulas — as
        a Bayesian prior belief about coefficients before seeing any data, or as an explicit
        bias-variance trade. Seeing both explains why the penalty terms look the way they do, instead
        of just accepting them as a formula to memorize.
      </P>

      <H2 c="The Bayesian View" />
      <P>
        Ordinary least squares is equivalent to Maximum Likelihood Estimation — finding the
        coefficients that make the observed data most probable, with no opinion about the
        coefficients themselves going in. <strong>Maximum A Posteriori (MAP)</strong> estimation adds
        a prior belief about what the coefficients probably look like <em>before</em> seeing the data,
        and regularization falls directly out of specific prior choices:
      </P>
      <Mx block>{`  L2 (Ridge)  ⟺  a Gaussian prior on each βⱼ:   βⱼ ~ N(0, τ²)
                  (a bell-curve belief: coefficients are probably small,
                   symmetric around 0, and large values are increasingly
                   unlikely but never impossible)

  L1 (Lasso)  ⟺  a Laplace prior on each βⱼ:     βⱼ ~ Laplace(0, b)
                  (a sharper peak exactly at 0, with heavier tails than
                   the Gaussian — this concentrated peak at zero is
                   exactly what produces sparsity: the prior actively
                   believes many coefficients are EXACTLY zero, not just
                   small)`}</Mx>
      <P>
        Maximizing the posterior probability (likelihood × prior) is mathematically identical to
        minimizing (negative log-likelihood + negative log-prior) — and the negative log of a Gaussian
        prior is a squared term, while the negative log of a Laplace prior is an absolute-value term.
        Ridge and Lasso's penalty formulas are exactly those two terms, with <Mx>λ</Mx> controlling how
        much the prior belief is trusted relative to the observed data.
      </P>

      <H2 c="The Bias-Variance Trade" />
      <P>
        Shrinking coefficients toward zero necessarily introduces some bias — the model can no longer
        fit the training data as tightly as an unconstrained fit could. In exchange, it reduces
        variance: a heavily regularized model changes much less from one training sample to another,
        because it's less able to chase noise specific to whatever data it happened to see. This is
        the exact same <Mx>Err = σ² + Bias² + Variance</Mx> decomposition from Validation &
        Bias-Variance — regularization is a direct, deliberate knob on that trade, not a separate idea.
      </P>
      <Note color="info" icon="ti-arrow-right">
        Since increasing λ always increases bias and decreases variance, picking the right λ is itself
        a model-selection problem best solved the same way any other hyperparameter is — by
        cross-validation, covered next under Hyperparameter Tuning.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.linear_model import RidgeCV, LassoCV

# Both search a range of lambda (alpha in sklearn) via cross-validation
# and refit the final model with the best one found.
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(X_train, y_train)
lasso = LassoCV(cv=5).fit(X_train, y_train)
print("Best lambda:", ridge.alpha_, lasso.alpha_)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="How does viewing Ridge regression as MAP estimation with a Gaussian prior explain why it never produces exact zeros?"
          a="A Gaussian prior on each coefficient believes values near zero are likely but treats every value as possible with smoothly decreasing probability — nothing in that belief singles out exactly zero as special. The resulting penalty (squared coefficients) is smooth everywhere, so the optimization has no special incentive to land exactly on zero rather than merely close to it." />
      <QA q="Why does a Laplace prior specifically, rather than any other 'prefers small values' prior, produce sparsity?"
          a="The Laplace distribution has a sharp peak exactly at zero — much more concentrated there than a Gaussian of similar spread — which translates into a penalty term with a non-differentiable kink at zero. That kink is what creates a real 'corner' in the optimization landscape that solutions frequently land on exactly, rather than merely near." />
      <QA q="If regularization increases bias, why would you ever want more of it?"
          a="Because total expected error is bias² plus variance, not bias alone — when a model is overfitting (variance is the dominant problem), trading a small amount of additional bias for a larger reduction in variance lowers total error even though bias technically went up. Regularization is worth using exactly when that trade is favorable, and harmful when the model was already underfitting (already high bias, low variance)." />
      <QA q="Does it make sense to apply regularization to a model that's currently underfitting the training data?"
          a="No — regularization pushes further in the direction of simpler, more constrained coefficients, which increases bias. A model that's already underfitting has too much bias already; adding regularization on top makes that worse. Regularization is the right tool specifically for high-variance, overfitting situations, not high-bias ones." />
    </div>
  );
}

const SECTION_MAP = {
  penalties: <SectionPenalties />,
  theory: <SectionTheory />,
};

export default function Regularization() {
  const [active, setActive] = useState('penalties');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 17</div>
        <h1 className="page-header-title">Regularization</h1>
        <p className="page-header-subtitle">
          Penalizing model complexity to fight overfitting — the deeper dive behind the Ridge/Lasso
          mentioned in Linear Models.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={17} />
    </div>
  );
}
