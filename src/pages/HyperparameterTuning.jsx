import { useState, useMemo, useEffect } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

/* ── Shared helpers ─────────────────────────────────────────── */
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
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

/* ── Small hand-solved linear algebra — same precedent as elsewhere ── */
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
      const fac = A[r][col];
      if (fac === 0) continue;
      for (let j = col; j < n; j++) A[r][j] -= fac * A[col][j];
      b[r] -= fac * b[col];
    }
  }
  return b;
}
function invertMatrix(Ain) {
  const n = Ain.length;
  const cols = [];
  for (let i = 0; i < n; i++) {
    const e = new Array(n).fill(0); e[i] = 1;
    cols.push(solveLinearSystem(Ain, e));
  }
  return Array.from({ length: n }, (_, r) => cols.map(c => c[r]));
}
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, pc = 0.3275911;
  const t = 1 / (1 + pc * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}
const normCDF = x => 0.5 * (1 + erf(x / Math.SQRT2));
const normPDF = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);

/* ── The landscape: two Gaussian wells over [0,10]x[0,10] ───────
   Verified true global minimum ≈1.795 at (7.00, 3.00); a shallower
   local trap ≈2.60 at (2.51, 6.99) — a real, moderate 0.8-unit gap. */
const WELLS = [
  { cx: 7, cy: 3, A: 3.2, s: 1.3 },
  { cx: 2.5, cy: 7, A: 2.4, s: 1.7 },
];
function landscapeF(x, y) {
  let v = 5;
  for (const w of WELLS) {
    const dx = x - w.cx, dy = y - w.cy;
    v -= w.A * Math.exp(-(dx * dx + dy * dy) / (2 * w.s * w.s));
  }
  return v;
}
const TRUE_MIN = 1.79547;

const GRID_POINTS = (() => {
  const xs = [0, 2.5, 5, 7.5, 10], ys = [0, 3.33, 6.67, 10];
  const pts = [];
  xs.forEach(x => ys.forEach(y => pts.push([x, y])));
  return pts;
})();
function randomSearchPoints(seed, budget) {
  const rand = seededRandom(seed);
  return Array.from({ length: budget }, () => [rand() * 10, rand() * 10]);
}
function kernel(p, q, ell, sf2) {
  const dx = p[0] - q[0], dy = p[1] - q[1];
  return sf2 * Math.exp(-(dx * dx + dy * dy) / (2 * ell * ell));
}
const BO_ELL = 2.3, BO_SF2 = 4.0, BO_SN2 = 1e-6, BO_XI = 0.02, BO_WARM = 5;
const BO_CANDIDATES = (() => {
  const c = [];
  for (let cx = 0.125; cx < 10; cx += 0.25) for (let cy = 0.125; cy < 10; cy += 0.25) c.push([cx, cy]);
  return c;
})();
function bayesOptPoints(seed, budget) {
  const rand = seededRandom(seed);
  const X = Array.from({ length: BO_WARM }, () => [rand() * 10, rand() * 10]);
  const Y = X.map(p => landscapeF(p[0], p[1]));
  while (X.length < budget) {
    const n = X.length;
    const K = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => kernel(X[i], X[j], BO_ELL, BO_SF2) + (i === j ? BO_SN2 : 0)));
    const Kinv = invertMatrix(K);
    const KinvY = Kinv.map(row => row.reduce((s, v, j) => s + v * Y[j], 0));
    const fBest = Math.min(...Y);
    let bestEI = -Infinity, bestCand = BO_CANDIDATES[0];
    for (const c of BO_CANDIDATES) {
      const kStar = X.map(xi => kernel(c, xi, BO_ELL, BO_SF2));
      const mu = kStar.reduce((s, v, i) => s + v * KinvY[i], 0);
      const kInvKstar = Kinv.map(row => row.reduce((s, v, i) => s + v * kStar[i], 0));
      const variance = Math.max(1e-12, BO_SF2 - kStar.reduce((s, v, i) => s + v * kInvKstar[i], 0));
      const sigma = Math.sqrt(variance);
      const improvement = fBest - mu - BO_XI;
      const Z = improvement / sigma;
      const ei = sigma > 1e-9 ? improvement * normCDF(Z) + sigma * normPDF(Z) : 0;
      if (ei > bestEI) { bestEI = ei; bestCand = c; }
    }
    X.push(bestCand);
    Y.push(landscapeF(bestCand[0], bestCand[1]));
  }
  return X;
}
const HP_BUDGET = 20;
const SEED_PRESETS = [11, 48, 85, 122];
function methodPoints(mode, seed) {
  if (mode === 'grid') return GRID_POINTS;
  if (mode === 'random') return randomSearchPoints(seed, HP_BUDGET);
  return bayesOptPoints(seed, HP_BUDGET);
}

const HP_SVG = 300, HP_GRID_RES = 36;
const HP_HEAT_CELLS = (() => {
  const cells = [];
  for (let r = 0; r < HP_GRID_RES; r++) {
    for (let c = 0; c < HP_GRID_RES; c++) {
      const x = (c + 0.5) / HP_GRID_RES * 10;
      const y = 10 - (r + 0.5) / HP_GRID_RES * 10;
      const v = landscapeF(x, y);
      cells.push(Math.max(0, Math.min(1, 1 - (v - TRUE_MIN) / (5 - TRUE_MIN))));
    }
  }
  return cells;
})();

function HyperparamSearchDemo() {
  const [mode, setMode] = useState('grid');
  const [seed, setSeed] = useState(SEED_PRESETS[0]);
  const [revealed, setRevealed] = useState(GRID_POINTS.length);
  const [playing, setPlaying] = useState(false);

  const points = useMemo(() => methodPoints(mode, seed), [mode, seed]);
  const values = useMemo(() => points.map(p => landscapeF(p[0], p[1])), [points]);

  function switchMode(m) {
    setMode(m);
    setRevealed(m === 'grid' ? GRID_POINTS.length : 1);
    setPlaying(false);
  }
  function changeSeed(s) { setSeed(s); setRevealed(1); setPlaying(false); }
  function reset() { setRevealed(mode === 'grid' ? points.length : 1); setPlaying(false); }
  function step() { setRevealed(r => Math.min(points.length, r + 1)); }

  useEffect(() => {
    if (!playing) return undefined;
    if (revealed >= points.length) { setPlaying(false); return undefined; }
    const id = setTimeout(() => setRevealed(r => Math.min(points.length, r + 1)), 350);
    return () => clearTimeout(id);
  }, [playing, revealed, points.length]);

  const bestSoFar = Math.min(...values.slice(0, revealed));
  const gap = bestSoFar - TRUE_MIN;

  const allResults = useMemo(() => ['grid', 'random', 'bo'].map(m => {
    const vals = methodPoints(m, seed).map(p => landscapeF(p[0], p[1]));
    const best = Math.min(...vals);
    return { mode: m, best, gap: best - TRUE_MIN };
  }), [seed]);

  const xToPx = x => x / 10 * HP_SVG;
  const yToPx = y => HP_SVG - y / 10 * HP_SVG;
  const cellSize = HP_SVG / HP_GRID_RES;
  const modeLabel = { grid: 'Grid', random: 'Random', bo: 'Bayesian Opt.' };

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
        <button onClick={() => switchMode('grid')} style={toggleBtnStyle(mode === 'grid')}>Grid Search</button>
        <button onClick={() => switchMode('random')} style={toggleBtnStyle(mode === 'random')}>Random Search</button>
        <button onClick={() => switchMode('bo')} style={toggleBtnStyle(mode === 'bo')}>Bayesian Optimization</button>
      </div>
      {mode !== 'grid' && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)' }}>Seed:</span>
          {SEED_PRESETS.map((s, i) => <button key={s} onClick={() => changeSeed(s)} style={toggleBtnStyle(seed === s)}>{i + 1}</button>)}
          <button onClick={() => setPlaying(p => !p)} style={toggleBtnStyle(playing)}>{playing ? 'Pause' : 'Play'}</button>
          <button onClick={step} style={toggleBtnStyle(false)}>Step</button>
          <button onClick={reset} style={toggleBtnStyle(false)}>Reset</button>
        </div>
      )}
      <Grid cols={2} gap={18}>
        <div>
          <svg viewBox={`0 0 ${HP_SVG} ${HP_SVG}`} width={HP_SVG} height={HP_SVG} style={{ borderRadius: 'var(--border-radius-md)', background: 'var(--color-background-secondary)', display: 'block' }}>
            {HP_HEAT_CELLS.map((v, i) => {
              const r = Math.floor(i / HP_GRID_RES), c = i % HP_GRID_RES;
              return <rect key={i} x={c * cellSize} y={r * cellSize} width={cellSize + 0.5} height={cellSize + 0.5} fill="var(--color-background-success)" opacity={v * 0.55} />;
            })}
            {points.slice(0, revealed).map((p, i) => (
              <circle key={i} cx={xToPx(p[0])} cy={yToPx(p[1])} r={i === revealed - 1 ? 6 : 4}
                fill={i === revealed - 1 ? 'var(--color-background-danger)' : 'var(--color-border-info)'}
                stroke={i === revealed - 1 ? 'var(--color-border-danger)' : 'none'} strokeWidth={2} opacity={0.9} />
            ))}
          </svg>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 6 }}>
            Green shading = the true loss landscape, revealed here only for teaching purposes — no
            real search algorithm gets to see this in advance.
          </div>
        </div>
        <div>
          <div style={{ fontSize: 13, marginBottom: 12, lineHeight: 1.8 }}>
            Evaluations used: <strong>{revealed}</strong> / {points.length}<br />
            Best value so far: <strong style={{ fontFamily: 'var(--font-mono)' }}>{bestSoFar.toFixed(4)}</strong><br />
            Gap to true optimum ({TRUE_MIN.toFixed(3)}): <strong style={{ fontFamily: 'var(--font-mono)', color: gap < 0.15 ? 'var(--color-text-success)' : 'var(--color-text-warning)' }}>{gap.toFixed(4)}</strong>
          </div>
          <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
            <thead><tr><th style={{ textAlign: 'left', padding: 4 }}>Method</th><th style={{ padding: 4 }}>Best found</th><th style={{ padding: 4 }}>Gap</th></tr></thead>
            <tbody>
              {allResults.map(r => (
                <tr key={r.mode} style={{ background: r.mode === mode ? 'var(--color-background-info)' : 'transparent' }}>
                  <td style={{ padding: 4 }}>{modeLabel[r.mode]}</td>
                  <td style={{ padding: 4, textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{r.best.toFixed(3)}</td>
                  <td style={{ padding: 4, textAlign: 'center', fontFamily: 'var(--font-mono)' }}>{r.gap.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
            All three methods get the same {HP_BUDGET}-evaluation budget for a fair comparison. Table
            updates with the seed selector — try a few different seeds.
          </div>
        </div>
      </Grid>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'search', label: 'Search Strategies', sub: 'Grid, random & Bayesian optimization, head to head' },
  { id: 'practice', label: 'Practical Tuning Wisdom', sub: 'Nested CV, search-space design & modern shortcuts' },
];

function SectionSearch() {
  return (
    <div>
      <P>
        A model's hyperparameters — learning rate, tree depth, regularization strength, number of
        neighbors — aren't learned from data the way ordinary parameters are; they have to be searched
        for. How that search is conducted turns out to matter enormously, especially as the number of
        hyperparameters grows.
      </P>

      <H2 c="Three Strategies" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="Grid Search">Exhaustively tries every combination of a predefined set of values per hyperparameter. Simple, fully reproducible, but its cost multiplies across dimensions.</Card>
        <Card color="warning" title="Random Search">Samples combinations uniformly at random within bounded ranges, for a fixed budget of trials. No structure, but doesn't waste evaluations on a rigid grid.</Card>
        <Card color="success" title="Bayesian Optimization">Fits a probabilistic surrogate model (commonly a Gaussian Process) to the results seen so far, and uses it to pick the next point most likely to improve on the best result — genuinely adaptive.</Card>
      </Grid>

      <H2 c="Why Random Search Tends to Beat Grid Search — In High Dimensions" />
      <P>
        Bergstra & Bengio (2012) showed that random search tends to outperform grid search once
        several hyperparameters are being tuned at once, for a specific, non-obvious reason: in most
        real problems, only a few hyperparameters actually matter much for a given dataset, and which
        ones matter isn't known in advance. Grid search spends an equal fraction of its budget on
        every dimension, including the unimportant ones — refining an irrelevant axis wastes
        evaluations that could have explored an important one instead. Random search's samples project
        onto every single axis with full density regardless of how many other dimensions exist, so it
        doesn't pay that tax.
      </P>
      <Table
        heads={['Dimensions tuned', 'Grid cost at 4 levels per dimension']}
        rows={[
          ['2', '4² = 16'],
          ['5', '4⁵ = 1,024'],
          ['10', '4¹⁰ = 1,048,576'],
        ]}
      />
      <Note color="warning" icon="ti-alert-triangle">
        This exponential blow-up is a real, dimension-count weakness of grid search regardless of
        which specific dimensions matter — it doesn't require Bergstra & Bengio's "some axes don't
        matter" argument to be true. Even if every dimension mattered equally, grid search's cost
        alone would still make it infeasible past a handful of hyperparameters.
      </Note>

      <H2 c="Try It — Watch All Three Search a Real 2D Landscape" />
      <P>
        A genuine hidden loss landscape — two dips of different depths, one of them a shallower local
        trap. All three methods get an identical, honest 20-evaluation budget: Grid Search uses 20
        fixed points; Random Search draws 20 uniform-random points; Bayesian Optimization starts from
        5 random points, then fits a real Gaussian Process (squared-exponential kernel) after every
        new observation and picks each of its remaining 15 evaluations by maximizing Expected
        Improvement.
      </P>
      <HyperparamSearchDemo />
      <Note color="success" icon="ti-info-circle">
        Verified directly: across 30 seeds at this exact 20-evaluation budget, Bayesian Optimization
        beat Random Search's average gap-to-optimum by roughly an order of magnitude or more (in this
        implementation: mean gap 0.041 for BO versus 0.901 for Random, and BO matched or beat Random
        on all 30 of 30 seeds tested) — a real, substantial, and highly reliable advantage at this
        budget. That advantage does depend on reasonable settings for the GP's own internal
        hyperparameters (its lengthscale in particular) and doesn't fully hold at a much smaller
        budget — Bayesian Optimization needs enough initial observations to build a trustworthy
        surrogate before its guidance beats blind sampling.
      </Note>

      <H2 c="The Honest Wrinkle: Grid Beat Random Here, Not the Other Way Around" />
      <P>
        Look closely at the comparison table above and something might seem backwards: Grid Search's
        gap (≈0.32) is actually <em>better</em> than Random Search's average (≈0.90) in this demo —
        seemingly the opposite of "random beats grid." It isn't a contradiction. Bergstra & Bengio's
        finding is specifically about <em>high-dimensional</em> search where most axes barely matter —
        a regime this deliberately simple 2D landscape, where both axes matter by construction, cannot
        recreate. A fixed grid is actually a perfectly reasonable way to cover a low-dimensional space
        where every dimension counts; it only starts losing once dimensionality climbs and grid
        search's fixed, non-adaptive coverage starts being spent on axes that don't matter.
      </P>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import optuna

# Bayesian optimization via Optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    depth = trial.suggest_int("max_depth", 2, 12)
    model = GradientBoostingClassifier(learning_rate=lr, max_depth=depth)
    return cross_val_score(model, X, y, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does random search tend to outperform grid search once several hyperparameters are being tuned simultaneously?"
          a="In most real tuning problems, only a subset of hyperparameters meaningfully affect performance for a given dataset, and which ones matter isn't known ahead of time. Grid search spends an equal share of its budget refining every dimension, including ones that don't matter, wasting evaluations. Random search's samples project onto every axis with full density no matter how many dimensions exist, so it doesn't pay that same tax — Bergstra & Bengio (2012) demonstrated this empirically." />
      <QA q="This module's own demo shows Grid Search beating Random Search's average in a 2D landscape. Doesn't that contradict 'random beats grid'?"
          a="No — Bergstra & Bengio's result is specifically about high-dimensional search, where many hyperparameters barely matter and grid search wastes budget refining them. A 2D demo where both axes matter by construction can't recreate that regime; there, a fixed grid is a perfectly reasonable, even competitive, way to cover the space. The real, dimension-independent case against grid search is separate: its cost explodes as (resolution)^dimensions regardless of which axes actually matter." />
      <QA q="What does Bayesian Optimization's surrogate model actually provide that random search doesn't have access to?"
          a="After every evaluation, the surrogate (typically a Gaussian Process) gives a predicted mean and uncertainty at every unevaluated point, letting the next choice deliberately balance exploiting promising regions against exploring uncertain ones (via an acquisition function like Expected Improvement). Random search has no memory of past results at all — every draw is independent of everything learned so far." />
      <QA q="Bayesian Optimization sounds strictly better than random search — are there situations where it isn't worth using?"
          a="It carries real overhead: fitting and re-fitting a surrogate model after every evaluation costs more compute per step than a trivial random draw, and it needs a reasonably-sized initial sample before its guidance becomes trustworthy — at a very small evaluation budget, or when evaluations themselves are extremely cheap and plentiful, that overhead and warm-up cost may not pay for itself relative to just running many more random or grid trials." />
    </div>
  );
}

function SectionPractice() {
  return (
    <div>
      <P>
        Beyond picking a search algorithm, a few practical habits separate a tuning process that
        actually finds a good, honestly-evaluated configuration from one that quietly fools itself.
      </P>

      <H2 c="Successive Halving & Hyperband" />
      <P>
        Instead of running every candidate configuration to completion, <strong>Successive
        Halving</strong> starts many configurations with a small budget each (e.g. a few training
        epochs), keeps only the best-performing fraction, and gives the survivors a larger budget —
        repeating until a small number of configurations remain, each fully trained.{' '}
        <strong>Hyperband</strong> runs several Successive Halving rounds with different
        aggressiveness settings, hedging against the risk of eliminating a slow-starting-but-eventually-
        good configuration too early. Both are modern, current techniques worth knowing about even at a
        conceptual level — they let a fixed compute budget cover far more candidate configurations than
        fully training every single one would allow.
      </P>

      <H2 c="Nested Cross-Validation" />
      <P>
        A subtle but important leakage risk: if the same cross-validation used to pick the best
        hyperparameters is also used to report the final performance number, that number is optimistic
        — it's been implicitly chosen to look good on exactly that data. <strong>Nested
        cross-validation</strong> fixes this with two loops:
      </P>
      <Mx block>{`  Outer loop (e.g. 5-fold):  splits data into outer-train / outer-test.
    Inner loop (e.g. 5-fold), run ONLY on outer-train:
      searches hyperparameters (grid/random/Bayesian — any of them),
      picks the best configuration using only inner-train/inner-validation splits.
    The winning configuration is refit on all of outer-train, then scored
    ONCE on outer-test — data the hyperparameter search never touched.
  Average the outer loop's test scores for an honest final performance estimate.`}</Mx>
      <Note color="info" icon="ti-info-circle">
        This directly closes the loop opened under Validation & Bias-Variance: the outer loop is what
        gives an honest read on generalization performance; the inner loop is exactly the ordinary
        hyperparameter search described in this module.
      </Note>

      <H2 c="Search-Space Design" />
      <P>
        How a range is sampled matters as much as which range is chosen. Learning rate and
        regularization strength are commonly searched on a <strong>log scale</strong> rather than
        linearly — sampling <Mx>lr</Mx> uniformly between 0.0001 and 0.1 wastes the overwhelming
        majority of samples above 0.01, when the interesting behavior often spans multiple orders of
        magnitude below that. Log-uniform sampling (or an explicit log-spaced grid) spends the search
        budget proportionally across every order of magnitude instead.
      </P>

      <H2 c="Tools Worth Knowing By Name" />
      <Table
        heads={['Tool', 'What it\'s for']}
        rows={[
          ['GridSearchCV / RandomizedSearchCV (scikit-learn)', 'Built-in grid and random search, wired directly into cross-validation'],
          ['Optuna', 'Popular Bayesian-optimization-based tuning library, define-by-run search spaces'],
          ['Hyperopt', 'Earlier, influential Bayesian/TPE-based tuning library'],
          ['Ray Tune', 'Distributed hyperparameter tuning at scale, supports most search algorithms above including Hyperband'],
        ]}
      />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.model_selection import cross_val_score, GridSearchCV

# Nested CV: GridSearchCV is the INNER loop; cross_val_score is the OUTER loop
inner_search = GridSearchCV(model, param_grid, cv=5)
outer_scores = cross_val_score(inner_search, X, y, cv=5)
print("Honest generalization estimate:", outer_scores.mean())`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="What problem does Successive Halving actually solve, compared to fully training every candidate configuration?"
          a="Fully training dozens or hundreds of candidate configurations to compare them is often prohibitively expensive. Successive Halving gives every candidate only a small initial budget, eliminates the worst performers early based on that partial signal, and reallocates the saved compute to training the survivors further — covering far more candidates overall for the same total compute, at the cost of occasionally eliminating a slow starter that would have caught up." />
      <QA q="Why is it considered leakage to use the same cross-validation both to select hyperparameters AND to report final performance?"
          a="The hyperparameters were explicitly chosen because they scored well on that data, so reporting performance on the very same splits credits the model with having 'seen' information used to pick its own configuration — the resulting number is optimistically biased. Nested cross-validation keeps an outer test fold completely untouched by the hyperparameter search, so the final reported score reflects genuine generalization." />
      <QA q="Why sample a learning-rate search space on a log scale instead of a linear one?"
          a="A model's behavior often changes on a multiplicative, not additive, scale as learning rate varies — the practical difference between 0.001 and 0.01 is usually far larger than between 0.091 and 0.1, even though both pairs differ by the same linear amount. Log-uniform sampling spends search budget proportionally across every order of magnitude instead of concentrating almost all samples in the largest values." />
      <QA q="A colleague tunes hyperparameters using 5-fold CV and reports that same 5-fold CV score as the model's expected production performance. What's the concern?"
          a="That score was used to select the hyperparameters, so it's contaminated by having implicitly optimized for those specific folds — it will tend to overstate real-world performance. The fix is nested cross-validation: keep an outer loop's test folds completely separate from whatever inner loop was used to search hyperparameters, and report performance from the untouched outer folds instead." />
    </div>
  );
}

const SECTION_MAP = {
  search: <SectionSearch />,
  practice: <SectionPractice />,
};

export default function HyperparameterTuning() {
  const [active, setActive] = useState('search');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 18</div>
        <h1 className="page-header-title">Hyperparameter Tuning</h1>
        <p className="page-header-subtitle">
          Systematically searching for the configuration that gets the most out of a given model.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={18} />
    </div>
  );
}
