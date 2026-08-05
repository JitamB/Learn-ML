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
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}
const GRID_COLS = 20, GRID_ROWS = 14;

/* ── CART fit (duplicated from the KNN & Decision Trees module — no shared util between pages) ── */
function giniOf(c1, n) {
  if (n === 0) return 0;
  const p1 = c1 / n;
  return 1 - p1 * p1 - (1 - p1) * (1 - p1);
}
function bestSplit(points, minLeaf) {
  const n = points.length;
  const totalOnes = points.filter(p => p.label === 1).length;
  let best = null;
  for (const feat of ['x', 'y']) {
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
function fitTree(points, depth, maxDepth, minLeaf) {
  const c1 = points.filter(p => p.label === 1).length;
  const g = giniOf(c1, points.length);
  const node = { label: c1 * 2 >= points.length ? 1 : 0, isLeaf: true };
  if (depth >= maxDepth || g === 0 || points.length < 2 * minLeaf) return node;
  const s = bestSplit(points, minLeaf);
  if (!s || s.w >= g - 1e-9) return node;
  node.isLeaf = false;
  node.feature = s.feat;
  node.threshold = s.thresh;
  node.left = fitTree(points.filter(p => p[s.feat] <= s.thresh), depth + 1, maxDepth, minLeaf);
  node.right = fitTree(points.filter(p => p[s.feat] > s.thresh), depth + 1, maxDepth, minLeaf);
  return node;
}
function predictTree(node, x, y) {
  if (node.isLeaf) return node.label;
  const v = node.feature === 'x' ? x : y;
  return v <= node.threshold ? predictTree(node.left, x, y) : predictTree(node.right, x, y);
}

/* ── Random Forest demo ─────────────────────────────────────── */
const BAGGING_POOL = (() => {
  const rand = seededRandom(9);
  return Array.from({ length: 56 }, () => {
    const x = rand() * 10, y = rand() * 10, noise = (rand() - 0.5) * 6; // noisier than the single-tree demo —
    return { x, y, label: (x + y + noise > 10) ? 1 : 0 };               // bagging needs real per-tree instability to smooth out
  });
})();
function bootstrapSample(pool, seed) {
  const r = seededRandom(seed);
  return Array.from({ length: pool.length }, () => pool[Math.floor(r() * pool.length)]);
}
const FOREST_SIZE = 25;

function RandomForestDemo() {
  const [nEstimators, setNEstimators] = useState(1);
  // Fit all 25 (deep, unpruned) trees ONCE on mount — real Random Forests use deep base trees
  // specifically because bagging only cancels out variance, and a shallow tree has little to cancel.
  const allTrees = useMemo(
    () => Array.from({ length: FOREST_SIZE }, (_, i) => fitTree(bootstrapSample(BAGGING_POOL, 42 + i), 0, 8, 1)),
    []
  );
  const activeTrees = allTrees.slice(0, nEstimators);
  const vote = (x, y) => {
    const votes1 = activeTrees.filter(t => predictTree(t, x, y) === 1).length;
    return votes1 * 2 >= activeTrees.length ? 1 : 0;
  };

  const cells = [];
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      cells.push(vote((c + 0.5) / GRID_COLS * 10, 10 - (r + 0.5) / GRID_ROWS * 10));
    }
  }
  const ensembleAcc = BAGGING_POOL.filter(p => vote(p.x, p.y) === p.label).length / BAGGING_POOL.length * 100;

  return (
    <VizBox>
      <SliderRow label="n_estimators" min={1} max={FOREST_SIZE} step={1} value={nEstimators} onChange={setNEstimators} fmt={v => `${v}`} />
      <div style={{ position: 'relative', height: 220, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${GRID_ROWS}, 1fr)` }}>
          {cells.map((cls, i) => (
            <div key={i} style={{ background: cls ? 'var(--color-background-danger)' : 'var(--color-background-info)' }} />
          ))}
        </div>
        {BAGGING_POOL.map((p, i) => (
          <div key={i} style={{
            position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
            transform: 'translate(-50%,-50%)', fontSize: 12, fontWeight: 700, lineHeight: 1,
            color: p.label ? 'var(--color-text-danger)' : 'var(--color-text-info)',
          }}>
            {p.label ? '✕' : '○'}
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        Train accuracy with {nEstimators} tree{nEstimators === 1 ? '' : 's'}: {ensembleAcc.toFixed(0)}%. Each tree is fit on its own
        bootstrap resample, so at n_estimators=1 the boundary is jagged and idiosyncratic — averaging
        more trees smooths it gradually, a handful of cells at a time, not all at once.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'bagging', label: 'Bagging', sub: 'Bootstrap aggregating' },
  { id: 'rf', label: 'Random Forest', sub: 'Bagging + feature subsampling' },
];

function SectionBagging() {
  return (
    <div>
      <P>
        Bagging (<strong>B</strong>ootstrap <strong>Agg</strong>regat<strong>ing</strong>) trains many
        independent copies of a model on different random resamples of the training data, then
        combines their predictions — averaging for regression, majority vote for classification.
      </P>

      <H2 c="Bootstrap Sampling" />
      <Mx block>{`  For a training set of size n, a bootstrap sample draws n rows
  WITH replacement from it.

  Each bootstrap sample typically contains ~63.2% of the unique original
  rows — the rest are duplicates, and the ~36.8% left out entirely form
  that tree's "out-of-bag" (OOB) set, usable as a free validation split.`}</Mx>

      <H2 c="Why Averaging Reduces Variance" />
      <P>
        A single model trained on one particular sample of data is noisy — retrain it on a slightly
        different sample and its predictions shift. Averaging many such models, each seeing a
        different resample, cancels out that noise as long as the individual models' errors aren't
        all making the exact same mistake in the same direction.
      </P>
      <Note color="warning" icon="ti-alert-triangle">
        Bagging only helps when the base model has genuinely high variance (deep trees, for example).
        A base model that's already stable — like a shallow, heavily-pruned tree, or linear
        regression — has little variance for averaging to cancel out, so bagging barely moves its
        performance.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),   # deliberately unpruned — high variance to cancel out
    n_estimators=25,
    random_state=42,
)
bag.fit(X_train, y_train)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does bagging use deep, unpruned trees as its base model rather than shallow ones?"
          a="Bagging works by averaging away variance, and a shallow tree already has very little variance to begin with — it's stable but biased. A deep, unpruned tree is the opposite: low bias but high variance, changing a lot from one bootstrap sample to the next. Averaging many such unstable trees cancels out most of that variance while keeping the low bias, which is a much better trade than bagging an already-stable shallow tree." />
      <QA q="What is an out-of-bag (OOB) score, and why is it useful?"
          a="Each bootstrap sample leaves out roughly 36.8% of the original rows on average. A model's OOB score evaluates it (or the ensemble) only on the rows each tree never saw during its own training — giving a validation-like estimate of generalization performance without setting aside a separate validation split." />
      <QA q="Does bagging reduce bias, variance, or both?"
          a="Primarily variance. Averaging many models trained on resampled data smooths out the noise specific to any one sample, but it doesn't fix a base model that's systematically wrong in the same way on every sample — that's a bias problem, which bagging leaves largely untouched. Boosting, by contrast, is specifically designed to reduce bias." />
    </div>
  );
}

function SectionRandomForest() {
  return (
    <div>
      <P>
        Random Forest is bagging with one extra twist: at every single split, each tree is only
        allowed to consider a random subset of the features, not all of them.
      </P>

      <H2 c="What's Different From Plain Bagging" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Row sampling (from bagging)">Each tree trains on its own bootstrap resample of the rows.</Card>
        <Card color="success" title="Feature sampling (new)">At each split, only a random subset of columns is considered — decorrelating the trees from each other.</Card>
      </Grid>
      <Note color="info" icon="ti-info-circle">
        Without feature subsampling, if one feature is a very strong predictor, nearly every bagged
        tree would split on it near the root — making the trees highly correlated with each other,
        which caps how much averaging can help. Forcing each split to ignore most features most of
        the time forces the trees to diversify.
      </Note>

      <H2 c="Benefits" />
      <Grid cols={3} gap={10}>
        <Card color="purple" title="Robust to overfitting">Averaging many decorrelated deep trees keeps the low bias while taming the variance.</Card>
        <Card color="warning" title="Handles messy data well">Tree splits are unaffected by outliers or unscaled features; missing values can often be routed through splits without imputation.</Card>
        <Card color="danger" title="Built-in feature importance">How much each feature reduces impurity, averaged across all trees, gives a free ranking of predictive features.</Card>
      </Grid>

      <H2 c="Try It — Add Trees One at a Time" />
      <P>Same 56 (noisier) points — one volatile tree at n_estimators=1, smoothing out as more join the vote.</P>
      <RandomForestDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",   # the feature-subsampling knob
    oob_score=True,
    random_state=42,
)
rf.fit(X_train, y_train)
print("OOB score:", rf.oob_score_)
print("Feature importances:", rf.feature_importances_)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does Random Forest add feature subsampling on top of what plain bagging already does?"
          a="Bagging alone only decorrelates trees through row resampling, which is a fairly weak source of diversity if one or two features are much stronger predictors than the rest — most trees would still split on the same dominant feature near the root and end up highly correlated. Restricting each split to a random feature subset forces genuinely different trees, and averaging correlated errors helps far less than averaging independent ones." />
      <QA q="How is a Random Forest's feature importance computed, at a high level?"
          a="For each split in each tree, record how much that split reduced impurity (Gini or variance), weighted by how many samples reached that node. Averaging this reduction for a given feature across every split and every tree in the forest gives that feature's overall importance score." />
      <QA q="Random Forest is often called robust to outliers. Why?"
          a="A tree split only asks whether a feature is above or below a threshold — an outlier still just falls on one side of that threshold like any other point, rather than distorting a distance calculation or a fitted coefficient the way it would in KNN or linear regression. The split location itself can shift slightly, but a single extreme value rarely dominates the tree's structure the way it would a mean or a covariance matrix." />
    </div>
  );
}

const SECTION_MAP = {
  bagging: <SectionBagging />,
  rf: <SectionRandomForest />,
};

export default function BaggingRandomForest() {
  const [active, setActive] = useState('bagging');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 07</div>
        <h1 className="page-header-title">Bagging & Random Forest</h1>
        <p className="page-header-subtitle">
          Combine many independently-trained models on resampled data to cancel out each other's
          errors and reduce variance versus any single model.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={7} />
    </div>
  );
}
