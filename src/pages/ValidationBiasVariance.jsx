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

/* ── Small hand-solved linear algebra (Gauss-Jordan) — same "small,      ──
   trusted, from-scratch linear algebra" precedent as DimensionalityReduction's
   closed-form 2x2 covariance/eigen-angle demo, generalized to n x n here
   since polynomial regression needs an arbitrary number of coefficients. */
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
function transpose(M) { return M[0].map((_, j) => M.map(row => row[j])); }
function matMul(A, B) { return A.map(row => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0))); }
function matVec(A, v) { return A.map(row => row.reduce((s, a, i) => s + a * v[i], 0)); }
function leastSquares(X, y) { const Xt = transpose(X); return solveLinearSystem(matMul(Xt, X), matVec(Xt, y)); }

/* ── Demo 1: Plain vs. Stratified K-Fold ───────────────────────
   24 points, 20 majority + 4 minority. Pure combinatorics, no fitting —
   verified across 5 seeds that plain K-fold always leaves at least one
   fold with zero minority examples, stratified never does. */
const KFOLD_BASE_LABELS = [...Array(20).fill(0), ...Array(4).fill(1)];
function fisherYates(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
function plainFolds(labels, K) {
  const n = labels.length;
  const folds = Array.from({ length: K }, () => []);
  labels.forEach((lab, i) => folds[Math.floor(i / (n / K))].push(lab));
  return folds;
}
function stratifiedFolds(labels, K) {
  const folds = Array.from({ length: K }, () => []);
  const byClass = {};
  labels.forEach((lab, i) => { (byClass[lab] ??= []).push(i); });
  Object.values(byClass).forEach(idxs => {
    idxs.forEach((idx, k) => folds[k % K].push(labels[idx]));
  });
  return folds;
}

function KFoldDemo() {
  const [mode, setMode] = useState('plain');
  const [seed, setSeed] = useState(5);
  const shuffled = useMemo(() => fisherYates(KFOLD_BASE_LABELS, seededRandom(seed)), [seed]);
  const folds = useMemo(() => mode === 'plain' ? plainFolds(shuffled, 4) : stratifiedFolds(shuffled, 4), [mode, shuffled]);
  const zeroFolds = folds.filter(f => f.filter(x => x === 1).length === 0).length;

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 14, flexWrap: 'wrap' }}>
        <button onClick={() => setMode('plain')} style={toggleBtnStyle(mode === 'plain')}>Plain K-Fold</button>
        <button onClick={() => setMode('stratified')} style={toggleBtnStyle(mode === 'stratified')}>Stratified K-Fold</button>
        <button onClick={() => setSeed(s => s + 137)} style={toggleBtnStyle(false)}>Reshuffle</button>
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {folds.map((fold, i) => {
          const minorityCount = fold.filter(x => x === 1).length;
          return (
            <div key={i} style={{ border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', padding: 10, minWidth: 118 }}>
              <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Fold {i + 1}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 8, maxWidth: 96 }}>
                {fold.map((lab, j) => (
                  <div key={j} style={{ width: 14, height: 14, borderRadius: '50%', background: lab ? 'var(--color-background-danger)' : 'var(--color-background-info)' }} />
                ))}
              </div>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: minorityCount === 0 ? 'var(--color-text-danger)' : 'var(--color-text-primary)' }}>
                {minorityCount} minority / {fold.length}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 12 }}>
        {mode === 'plain'
          ? (zeroFolds > 0
            ? `${zeroFolds} of 4 folds have zero minority-class examples in this shuffle — that fold's minority-class precision and recall are literally undefined.`
            : `This particular shuffle happens to spread the minority class out — hit Reshuffle a few more times and watch that stop being true.`)
          : 'Stratified K-Fold assigns each class round-robin, independent of shuffle order — every fold gets exactly 1 of the 4 minority examples, every single time.'}
      </div>
    </VizBox>
  );
}

/* ── Demo 2: Polynomial-degree bias-variance slider ────────────
   Real closed-form least squares (normal equation) on polynomial features,
   degree 1-15. x is scaled to [-1,1] before building features — fitting
   raw x in [0,10] is severely ill-conditioned at high degree (verified:
   training MSE, which must be monotonically non-increasing, actually rose
   from degree 10->11 from pure float error without this scaling step). */
const trueFn = x => Math.sin(1.3 * x) * 3 + 0.3 * x;
const BV_N = 16, BV_MAX_DEGREE = 15;
const BV_TRAIN = (() => {
  const rand = seededRandom(12);
  return Array.from({ length: BV_N }, (_, i) => {
    const x = (i + 0.5) / BV_N * 10;
    return { x, y: trueFn(x) + randNormalish(rand, 0, 0.9) };
  });
})();
const BV_VAL = (() => {
  const rand = seededRandom(1012);
  return Array.from({ length: BV_N }, (_, i) => {
    const x = (i + 0.5) / BV_N * 10;
    return { x, y: trueFn(x) + randNormalish(rand, 0, 0.9) };
  });
})();
function bvDesign(xs, degree) {
  return xs.map(x => {
    const xs_ = (x - 5) / 5;
    const row = [1];
    let p = 1;
    for (let d = 1; d <= degree; d++) { p *= xs_; row.push(p); }
    return row;
  });
}
function bvPredict(beta, x) {
  const xs_ = (x - 5) / 5;
  let p = 1, yhat = beta[0];
  for (let d = 1; d < beta.length; d++) { p *= xs_; yhat += beta[d] * p; }
  return yhat;
}
function bvMSE(points, beta) { return points.reduce((s, p) => s + (p.y - bvPredict(beta, p.x)) ** 2, 0) / points.length; }
const BV_CURVE = (() => {
  const xs = BV_TRAIN.map(p => p.x), ys = BV_TRAIN.map(p => p.y);
  const out = [];
  for (let degree = 1; degree <= BV_MAX_DEGREE; degree++) {
    const beta = leastSquares(bvDesign(xs, degree), ys);
    out.push({ degree, beta, trainMSE: bvMSE(BV_TRAIN, beta), valMSE: bvMSE(BV_VAL, beta) });
  }
  return out;
})();
const BV_BEST = BV_CURVE.reduce((best, r) => r.valMSE < best.valMSE ? r : best, BV_CURVE[0]);
const BV_MAX_MSE = Math.max(...BV_CURVE.map(r => Math.max(r.trainMSE, r.valMSE)));

const BV_X_MIN = 0, BV_X_MAX = 10, BV_Y_MIN = -8, BV_Y_MAX = 16;
const BV_SVG_W = 300, BV_SVG_H = 220;
const bvXToPx = x => (x - BV_X_MIN) / (BV_X_MAX - BV_X_MIN) * BV_SVG_W;
const bvYToPx = y => BV_SVG_H - (y - BV_Y_MIN) / (BV_Y_MAX - BV_Y_MIN) * BV_SVG_H;

function PolynomialBiasVarianceDemo() {
  const [degree, setDegree] = useState(1);
  const current = BV_CURVE[degree - 1];

  const curvePath = useMemo(() => {
    const pts = [];
    for (let x = BV_X_MIN; x <= BV_X_MAX; x += 0.08) pts.push([bvXToPx(x), bvYToPx(bvPredict(current.beta, x))]);
    return 'M ' + pts.map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`).join(' L ');
  }, [current]);

  const mseW = 300, mseH = 130, msePadL = 30;
  const degToPx = d => msePadL + (d - 1) / (BV_MAX_DEGREE - 1) * (mseW - msePadL - 8);
  const mseToPx = v => mseH - 6 - Math.min(v, BV_MAX_MSE) / BV_MAX_MSE * (mseH - 16);
  const trainPath = 'M ' + BV_CURVE.map(r => `${degToPx(r.degree).toFixed(1)},${mseToPx(r.trainMSE).toFixed(1)}`).join(' L ');
  const valPath = 'M ' + BV_CURVE.map(r => `${degToPx(r.degree).toFixed(1)},${mseToPx(r.valMSE).toFixed(1)}`).join(' L ');

  return (
    <VizBox>
      <SliderRow label="Polynomial degree" min={1} max={BV_MAX_DEGREE} step={1} value={degree} onChange={setDegree} fmt={v => String(v)} />
      <Grid cols={2} gap={16}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>Fit at degree {degree}</div>
          <svg viewBox={`0 0 ${BV_SVG_W} ${BV_SVG_H}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
            <path d={curvePath} fill="none" stroke="var(--accent)" strokeWidth={2} />
            {BV_TRAIN.map((p, i) => <circle key={'t' + i} cx={bvXToPx(p.x)} cy={bvYToPx(p.y)} r={3.2} fill="var(--color-border-info)" />)}
            {BV_VAL.map((p, i) => <circle key={'v' + i} cx={bvXToPx(p.x)} cy={bvYToPx(p.y)} r={3.2} fill="var(--color-border-danger)" />)}
          </svg>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 4 }}>
            <span style={{ color: 'var(--color-text-info)' }}>● train</span> &nbsp; <span style={{ color: 'var(--color-text-danger)' }}>● validation</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 4 }}>Error vs. degree</div>
          <svg viewBox={`0 0 ${mseW} ${mseH}`} style={{ width: '100%', height: 'auto', display: 'block', background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
            <path d={trainPath} fill="none" stroke="var(--color-border-info)" strokeWidth={2} />
            <path d={valPath} fill="none" stroke="var(--color-border-danger)" strokeWidth={2} />
            <line x1={degToPx(degree)} y1={0} x2={degToPx(degree)} y2={mseH} stroke="var(--color-border-tertiary)" strokeWidth={1} strokeDasharray="3,3" />
            <circle cx={degToPx(BV_BEST.degree)} cy={mseToPx(BV_BEST.valMSE)} r={3.5} fill="var(--color-text-success)" />
          </svg>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 4 }}>
            <span style={{ color: 'var(--color-text-info)' }}>● train MSE</span> &nbsp; <span style={{ color: 'var(--color-text-danger)' }}>● val MSE</span> &nbsp; <span style={{ color: 'var(--color-text-success)' }}>● best</span>
          </div>
        </div>
      </Grid>
      <div style={{ display: 'flex', gap: 18, marginTop: 10, fontSize: 12.5, fontFamily: 'var(--font-mono)' }}>
        <span>train MSE = {current.trainMSE.toFixed(3)}</span>
        <span>val MSE = {current.valMSE.toFixed(3)}</span>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        Degree {BV_BEST.degree} minimizes validation error ({BV_BEST.valMSE.toFixed(3)}) — the honest sweet spot. Training
        error keeps falling all the way to {BV_CURVE[BV_MAX_DEGREE - 1].trainMSE.toFixed(4)} at degree {BV_MAX_DEGREE} (it
        exactly interpolates all {BV_N} training points), while validation error climbs back up to{' '}
        {BV_CURVE[BV_MAX_DEGREE - 1].valMSE.toFixed(3)} — worse than several much simpler degrees. Push the slider past
        ~degree 12 and watch the fitted curve start whipping off the chart between data points, even though it still
        threads every training point exactly.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'validation', label: 'Validation Strategies', sub: 'Splits, cross-validation & the pitfalls that invalidate them' },
  { id: 'biasvariance', label: 'Bias-Variance Tradeoff', sub: 'Why models underfit or overfit, formalized' },
];

function SectionValidation() {
  return (
    <div>
      <P>
        A model's training accuracy tells you almost nothing about how it will perform on data it
        hasn't seen — and data it hasn't seen is the only kind that matters once it's deployed.
        Validation is the discipline of measuring that honestly, and of never letting a number you
        trust get contaminated by information it shouldn't have had access to.
      </P>

      <H2 c="Train, Validation & Test — Why Three Sets, Not Two" />
      <P>
        <strong>Training set</strong>: what the model learns from. <strong>Validation set</strong>:
        what you use to make decisions — which model, which hyperparameters, when to stop training.
        <strong> Test set</strong>: touched exactly once, at the very end, to report a final number.
        The reason a validation set exists separately from the test set is that the moment you use a
        dataset to make even one decision (picking a hyperparameter, choosing between two model
        families), you've fit that decision to that data — reusing it afterward as your final,
        "honest" number is no longer honest. A common split for small-to-medium datasets is
        60/20/20 or 80/10/10; for very large datasets even 1% held out is often enough (1% of 10
        million rows is still 100,000 test examples).
      </P>

      <H2 c="K-Fold Cross-Validation" />
      <P>
        Instead of a single validation split, split the training data into <Mx>K</Mx> equal folds.
        Train <Mx>K</Mx> separate models, each time holding out a different fold as validation and
        training on the remaining <Mx>K−1</Mx>. Average the <Mx>K</Mx> validation scores. Every point
        eventually gets used for validation exactly once, which reduces the variance of the
        performance estimate itself compared to one lucky (or unlucky) single split.
      </P>
      <Mx block>{`  K = 5 or K = 10 is standard. Increasing K:
    + less bias in the performance estimate (each training fold is closer to
      the full dataset's size)
    − more variance in the estimate (folds overlap more, so the K scores
      become more correlated with each other) and more compute (K full
      training runs instead of 1)
  K = n (every fold has exactly 1 point) is Leave-One-Out CV — see below.`}</Mx>

      <H2 c="Stratified K-Fold" />
      <P>
        Plain K-Fold shuffles rows and cuts them into <Mx>K</Mx> contiguous-after-shuffling chunks —
        it has no idea what the class labels are, so on an imbalanced dataset it's entirely possible
        (and, as the demo below shows, actually likely) for a fold to end up with too few or even
        zero examples of the minority class. <strong>Stratified K-Fold</strong> fixes this by
        assigning each class to folds independently, so every fold's class balance mirrors the full
        dataset's.
      </P>

      <H2 c="Try It — Watch a Fold Lose Its Minority Class Entirely" />
      <P>
        24 points, 20 of one class and only 4 of another. Toggle between plain and stratified K-Fold
        (K=4) and reshuffle repeatedly.
      </P>
      <KFoldDemo />

      <H2 c="Leave-One-Out Cross-Validation (LOOCV)" />
      <P>
        The <Mx>K = n</Mx> special case: train on all but one point, validate on that one point,
        repeat for every point. Nearly unbiased (each training fold is as large as possible), but
        expensive (<Mx>n</Mx> full training runs) and, for many models, the <Mx>n</Mx> individual
        scores end up highly correlated with each other (each training fold differs from the next by
        only one point), which can make the aggregate estimate itself surprisingly high-variance
        despite each individual fit being near-unbiased. Still genuinely useful when <Mx>n</Mx> is
        very small — a 40-patient medical study can't spare 20% of its data for a single validation
        split, but can afford 40 quick refits.
      </P>

      <Note color="info" icon="ti-info-circle">
        Ordinary K-Fold assumes rows are exchangeable — shuffling them is safe. That assumption
        breaks for sequential data: shuffling a time series before splitting lets the model train on
        rows from the future and validate on rows from the past, silently leaking information no real
        deployment would ever have. The fix, <strong>walk-forward (or rolling-window) cross-
        validation</strong>, always trains on the past and validates on the future — covered in full
        once Time Series Analysis is built.
      </Note>
      <Note color="info" icon="ti-arrow-right">
        A related question — "how do I tune hyperparameters without leaking test-set information into
        that choice?" — is answered by <strong>Nested Cross-Validation</strong>, covered under
        Hyperparameter Tuning.
      </Note>

      <H2 c="Data Leakage — The Most Interview-Common Failure Mode Here" />
      <P>
        Leakage means information that shouldn't be available at prediction time somehow influenced
        training or the reported score — the result is a model that looks great on paper and fails
        quietly in production. It comes in several distinct, all-common flavors:
      </P>
      <Table
        heads={['Type', 'What happens', 'Concrete example']}
        rows={[
          ['Preprocessing leakage', 'Fitting a scaler, imputer, or PCA on the full dataset before splitting', 'A StandardScaler fit on train+test together means the training fold\'s mean/std were computed partly from test-set values'],
          ['Target leakage', 'A feature encodes the outcome itself', '"days_since_cancellation" as a feature for predicting churn'],
          ['Temporal leakage', 'A feature is only knowable after the prediction point', '"final account balance" used to predict "will they close the account this month"'],
          ['Group leakage', 'The same entity appears in both train and test', 'The same patient\'s scans split across train and test — fixed with GroupKFold'],
          ['Duplicate-row leakage', 'Near-identical rows land on both sides of the split', 'Inflates apparent performance without the model learning anything new'],
        ]}
      />
      <Note color="warning" icon="ti-alert-triangle">
        The fix for preprocessing leakage is always the same shape: fit any transformer
        (<code>StandardScaler</code>, an imputer, PCA) on the training fold only, then just
        <code>.transform()</code> — never <code>.fit()</code> — on validation/test data. "Spot the bug
        in this pipeline" questions built around exactly this mistake are extremely common.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline ensures the scaler is refit on each training fold only —
# no preprocessing leakage, even though cross_val_score handles the looping.
pipe = Pipeline([("scale", StandardScaler()), ("clf", model)])
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits=5))
print(scores.mean(), scores.std())`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why use K-Fold CV instead of a single train/validation split?"
          a="A single split gives one performance estimate that depends heavily on which points happened to land in validation — an unlucky split can make a good model look bad, or vice versa. K-Fold averages over K different splits, using every point for validation exactly once, which gives a more stable, lower-variance estimate of true performance, at the cost of K times the training compute." />
      <QA q="What's actually wrong with fitting a StandardScaler on the entire dataset before doing train_test_split?"
          a="The scaler's mean and standard deviation get computed using test-set values, so the 'training' data has indirectly seen information about the test set's distribution. It's a subtle form of preprocessing leakage — the reported test performance will be optimistically biased, sometimes only slightly, but the fix (fit the scaler only on the training fold, then transform the rest) costs nothing and should always be done regardless." />
      <QA q="When would you actually choose LOOCV over 5-fold or 10-fold CV, given it's far more expensive?"
          a="When the dataset is small enough that giving up even 20% of it for a single fold's validation set meaningfully starves the training folds — common in domains like clinical studies with only a few dozen samples. At that scale, n individual refits is affordable and LOOCV's near-unbiased estimate is worth the compute; at 100,000 rows, nobody runs LOOCV." />
      <QA q="Why can't you just shuffle a time-series dataset and run ordinary K-Fold CV on it?"
          a="Shuffling breaks the temporal order the data actually has — a model could end up training on rows from next month and validating on rows from last month, which is information no real deployed model would ever have at prediction time. Walk-forward (rolling-window) cross-validation fixes this by always training on the past and validating on a later, held-out period." />
    </div>
  );
}

function SectionBiasVariance() {
  return (
    <div>
      <P>
        Every model's error on new data breaks down into three additive pieces — one you can never
        remove, and two you trade off against each other by adjusting model complexity. Understanding
        this decomposition explains, in one shot, why simple models underfit and complex models
        overfit, and why "just make the model bigger" is not a universal fix.
      </P>

      <H2 c="The Formal Decomposition" />
      <P>
        Suppose the true relationship is <Mx>y = f(x) + ε</Mx>, where <Mx>ε</Mx> is irreducible noise
        with mean 0 and variance <Mx>σ²</Mx>. Train a model <Mx>f̂</Mx> on a random training set, and
        ask: what's the expected squared error at a point <Mx>x</Mx>, averaged over every possible
        training set you could have drawn?
      </P>
      <Mx block>{`  Err(x) = E[(y − f̂(x))²]
          = E[(f(x) + ε − f̂(x))²]
          = σ²  +  E[(f(x) − f̂(x))²]        (the ε cross-term vanishes: ε is
                                              independent of f̂ and averages to 0)

  Now let f̄(x) = E[f̂(x)]  (the average prediction, over many resampled
  training sets). Expanding around f̄(x):

          E[(f(x) − f̂(x))²] = (f(x) − f̄(x))²  +  E[(f̂(x) − f̄(x))²]
                             =     Bias(f̂(x))²   +   Var(f̂(x))
                             (this cross-term also vanishes, for the same reason)

  Err(x)  =  σ²  +  Bias(f̂(x))²  +  Var(f̂(x))
          irreducible    how wrong the      how much the
            noise      average prediction   prediction swings
                          is, systematically  across different
                                               training sets`}</Mx>
      <Note color="info" icon="ti-info-circle">
        <Mx>σ²</Mx> can't be reduced by any model choice — it's noise inherent to the data-generating
        process itself. Everything a model's complexity controls is the trade between the other two
        terms.
      </Note>

      <H2 c="Underfitting & Overfitting, Concretely" />
      <Grid cols={2} gap={10}>
        <Card color="warning" title="High Bias (Underfitting)">The model is too simple to capture the real pattern — a straight line fit to a curve. Training error is already high, because the model can't even represent the true relationship. As covered under KNN & Decision Trees: a large K in KNN, or a shallow decision tree, both underfit.</Card>
        <Card color="danger" title="High Variance (Overfitting)">The model is complex enough to fit noise in the training set, so it changes drastically depending on exactly which training points it saw. Training error is very low; validation error is much higher. A small K in KNN, or a very deep decision tree, both overfit — same footnote, same underlying tradeoff.</Card>
      </Grid>

      <H2 c="Learning Curves vs. Validation Curves" />
      <P>
        These get confused constantly, and they answer different questions:
      </P>
      <Table
        heads={['', 'X-axis', 'What it diagnoses']}
        rows={[
          ['Learning curve', 'Training set size', 'Would more data help? If train and validation error have both plateaued at a similar, high value, more data alone won\'t close the gap — the model itself is too simple (high bias). If there\'s still a persistent gap between them, more data (or regularization) can help.'],
          ['Validation curve', 'A complexity knob (polynomial degree, tree depth, K, regularization strength)', 'What\'s the right complexity for this model, on this amount of data? The demo below is exactly a validation curve.'],
        ]}
      />

      <H2 c="Try It — Sweep Model Complexity, Watch the U-Curve Form" />
      <P>
        A real polynomial regression, refit from scratch at every degree via closed-form least
        squares, on 16 noisy training points from a genuinely non-linear function. The right panel
        traces both curves as degree sweeps from 1 to 15 — it's a live validation curve.
      </P>
      <PolynomialBiasVarianceDemo />

      <Note color="warning" icon="ti-alert-triangle">
        <strong>Double descent</strong> (Belkin et al., 2019) is worth knowing about even though this
        demo doesn't show it: for very overparameterized models (far more parameters than training
        points — the regime deep networks usually live in, not this 16-point polynomial), test error
        can rise through the classical U-curve's peak and then <em>fall again</em> past the point where
        the model can exactly fit its training data. The classical bias-variance intuition above still
        explains the vast majority of everyday model-selection situations; double descent is a real,
        separate phenomenon specific to heavily overparameterized regimes, not a contradiction of it.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.model_selection import validation_curve
import numpy as np

degrees = np.arange(1, 16)
train_scores, val_scores = validation_curve(
    make_pipeline(PolynomialFeatures(), LinearRegression()),
    X, y, param_name="polynomialfeatures__degree", param_range=degrees,
    scoring="neg_mean_squared_error", cv=5,
)
# train_scores.mean(axis=1) keeps falling; val_scores.mean(axis=1) U-turns`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="A model has near-zero training error but much higher validation error. Is this high bias or high variance?"
          a="High variance. Near-zero training error means the model is expressive enough to fit (or memorize) the training data almost perfectly — that rules out high bias, which would show up as poor performance even on training data. The large train/validation gap is the signature of overfitting: the model learned patterns specific to the training sample that don't generalize." />
      <QA q="Why does the bias-variance decomposition include an irreducible error term, and can better modeling ever remove it?"
          a="It represents genuine randomness in how the target is generated given the features — even a model that predicted the true conditional mean f(x) perfectly would still miss by this amount on average, because real outcomes vary for reasons the features don't capture. No amount of better modeling reduces it; it's a property of the data-generating process, not of any particular model." />
      <QA q="You plot a learning curve and see training and validation error converge to a similar, high value as training set size grows. What should you do?"
          a="Collecting more data won't help much on its own — both curves have already plateaued, meaning the model has hit its bias ceiling. The fix is to increase model complexity (a richer feature set, a more expressive model family) or reduce bias some other way, not to gather a bigger training set." />
      <QA q="What's the practical difference between a learning curve and a validation curve, and when would you use each?"
          a="A learning curve varies training-set size to diagnose whether more data would help, holding the model fixed. A validation curve varies a complexity/hyperparameter knob at a fixed dataset size to diagnose the right amount of model complexity. They answer 'do I need more data?' versus 'do I need a different model?' respectively, and are easy to conflate because both plot an error curve." />
    </div>
  );
}

const SECTION_MAP = {
  validation: <SectionValidation />,
  biasvariance: <SectionBiasVariance />,
};

export default function ValidationBiasVariance() {
  const [active, setActive] = useState('validation');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 15</div>
        <h1 className="page-header-title">Validation & Bias-Variance</h1>
        <p className="page-header-subtitle">
          Getting an honest read on model performance, and understanding the bias-variance tradeoff
          that explains why models underfit or overfit in the first place.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={15} />
    </div>
  );
}
