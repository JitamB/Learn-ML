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
const clamp10 = v => Math.max(0.3, Math.min(9.7, v));

/* ── K-Nearest Neighbors demo ───────────────────────────────── */
function randNormalish(rand, mean, std) {
  const u = rand() + rand() + rand() - 1.5; // sum of 3 uniforms, range [-1.5, 1.5], bell-ish
  return mean + (u / 0.5) * std;
}
const KNN_POINTS = (() => {
  const randA = seededRandom(5), randB = seededRandom(31);
  const pts = [];
  for (let i = 0; i < 24; i++) pts.push({ x: clamp10(randNormalish(randA, 3.3, 2.0)), y: clamp10(randNormalish(randB, 7, 2.0)), label: 0 });
  for (let i = 0; i < 22; i++) pts.push({ x: clamp10(randNormalish(randA, 6.7, 2.0)), y: clamp10(randNormalish(randB, 3, 2.0)), label: 1 });
  const salted = [ // deliberately mislabeled, sitting deep in the opposite class's territory —
    { x: 2.2, y: 7.5, label: 1 }, { x: 3.0, y: 8.2, label: 1 }, { x: 1.6, y: 6.0, label: 1 }, // without these, K barely matters visually
    { x: 7.8, y: 2.5, label: 0 }, { x: 8.4, y: 3.6, label: 0 }, { x: 6.0, y: 1.6, label: 0 },
  ];
  return pts.concat(salted);
})();

function classifyKNN(px, py, k) {
  const withDist = KNN_POINTS
    .map(p => ({ d2: (p.x - px) ** 2 + (p.y - py) ** 2, label: p.label }))
    .sort((a, b) => a.d2 - b.d2);
  let c0 = 0, c1 = 0;
  for (let i = 0; i < k; i++) (withDist[i].label === 0 ? c0++ : c1++);
  return c0 > c1 ? 0 : 1;
}

function KNNDemo() {
  const [k, setK] = useState(5);
  const cells = [];
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const px = (c + 0.5) / GRID_COLS * 10;
      const py = 10 - (r + 0.5) / GRID_ROWS * 10;
      cells.push(classifyKNN(px, py, k));
    }
  }

  return (
    <VizBox>
      <SliderRow label="K" min={1} max={15} step={2} value={k} onChange={setK} fmt={v => `${v}`} />
      <div style={{ position: 'relative', height: 220, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${GRID_ROWS}, 1fr)` }}>
          {cells.map((cls, i) => (
            <div key={i} style={{ background: cls ? 'var(--color-background-danger)' : 'var(--color-background-info)' }} />
          ))}
        </div>
        {KNN_POINTS.map((p, i) => (
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
        {k === 1
          ? 'K=1: every point (even the deliberately mislabeled ones) gets its own little region — the classic overfitting signature.'
          : k >= 13
          ? 'K is now large enough that a single mislabeled point barely swings the vote — smooth, but starting to blur real local structure too.'
          : 'A few mislabeled points still carve out small islands, but the boundary is visibly smoothing out as K grows.'}
      </div>
    </VizBox>
  );
}

/* ── Decision Tree demo: real Gini-CART + tree diagram ──────── */
const TREE_POINTS = (() => {
  const rand = seededRandom(13);
  return Array.from({ length: 56 }, () => {
    const x = rand() * 10, y = rand() * 10, noise = (rand() - 0.5) * 1.0;
    return { x, y, label: (x + y + noise > 10) ? 1 : 0 };
  });
})();

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
  const node = { n: points.length, label: c1 * 2 >= points.length ? 1 : 0, gini: g, isLeaf: true, depth };
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
function layoutTree(root) {
  let nextSlot = 0;
  (function assignLeafSlots(node) {
    if (node.isLeaf) { node.xSlot = nextSlot++; return; }
    assignLeafSlots(node.left);
    assignLeafSlots(node.right);
  })(root);
  (function assignInternalSlots(node) {
    if (node.isLeaf) return node.xSlot;
    node.xSlot = (assignInternalSlots(node.left) + assignInternalSlots(node.right)) / 2;
    return node.xSlot;
  })(root);
  const numLeaves = Math.max(1, nextSlot);
  const maxD = Math.max(1, (function maxDepthOf(node) {
    return node.isLeaf ? node.depth : Math.max(maxDepthOf(node.left), maxDepthOf(node.right));
  })(root));
  const nodes = [], edges = [];
  (function collect(node) {
    node.xPct = (node.xSlot + 0.5) / numLeaves * 100;
    node.yPct = (node.depth / maxD) * 100;
    nodes.push(node);
    if (!node.isLeaf) {
      edges.push([node, node.left]);
      edges.push([node, node.right]);
      collect(node.left);
      collect(node.right);
    }
  })(root);
  return { nodes, edges };
}

const TREE_DIAGRAM_H = 220, TREE_DIAGRAM_PAD_Y = 20;
const treeNodeY = node => TREE_DIAGRAM_PAD_Y + (node.yPct / 100) * (TREE_DIAGRAM_H - 2 * TREE_DIAGRAM_PAD_Y);

function DecisionTreeDemo() {
  const [maxDepth, setMaxDepth] = useState(2);
  const tree = useMemo(() => fitTree(TREE_POINTS, 0, maxDepth, 2), [maxDepth]);
  const { nodes, edges } = useMemo(() => layoutTree(tree), [tree]);
  const trainAcc = useMemo(
    () => TREE_POINTS.filter(p => predictTree(tree, p.x, p.y) === p.label).length / TREE_POINTS.length * 100,
    [tree]
  );

  const cells = [];
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const px = (c + 0.5) / GRID_COLS * 10;
      const py = 10 - (r + 0.5) / GRID_ROWS * 10;
      cells.push(predictTree(tree, px, py));
    }
  }

  return (
    <VizBox>
      <SliderRow label="max_depth" min={1} max={4} step={1} value={maxDepth} onChange={setMaxDepth} fmt={v => `${v}`} />
      <Grid cols={2} gap={12}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Decision boundary</div>
          <div style={{ position: 'relative', height: TREE_DIAGRAM_H, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${GRID_ROWS}, 1fr)` }}>
              {cells.map((cls, i) => (
                <div key={i} style={{ background: cls ? 'var(--color-background-danger)' : 'var(--color-background-info)' }} />
              ))}
            </div>
            {TREE_POINTS.map((p, i) => (
              <div key={i} style={{
                position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
                transform: 'translate(-50%,-50%)', fontSize: 11, fontWeight: 700, lineHeight: 1,
                color: p.label ? 'var(--color-text-danger)' : 'var(--color-text-info)',
              }}>
                {p.label ? '✕' : '○'}
              </div>
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Tree structure ({nodes.length} nodes)</div>
          <div style={{ position: 'relative', height: TREE_DIAGRAM_H, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)' }}>
            <svg viewBox="0 0 100 220" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
              {edges.map(([parent, child], i) => (
                <line key={i} x1={parent.xPct} y1={treeNodeY(parent)} x2={child.xPct} y2={treeNodeY(child)}
                  stroke="var(--color-border-secondary)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
              ))}
            </svg>
            {nodes.map((node, i) => (
              <div key={i} style={{
                position: 'absolute', left: `${node.xPct}%`, top: treeNodeY(node),
                transform: 'translate(-50%, -50%)', maxWidth: 74,
                padding: '3px 5px', borderRadius: 'var(--border-radius-sm)', fontSize: 9.5, fontFamily: 'var(--font-mono)',
                whiteSpace: 'nowrap', textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis',
                border: '1px solid ' + (node.isLeaf ? 'var(--color-border-tertiary)' : 'var(--color-border-info)'),
                background: node.isLeaf ? (node.label ? 'var(--color-background-danger)' : 'var(--color-background-info)') : 'var(--color-background-primary)',
                color: node.isLeaf ? (node.label ? 'var(--color-text-danger)' : 'var(--color-text-info)') : 'var(--color-text-primary)',
              }}>
                {node.isLeaf ? `class ${node.label}` : `${node.feature} ≤ ${node.threshold.toFixed(1)}`}
              </div>
            ))}
          </div>
        </div>
      </Grid>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Train accuracy at depth {maxDepth}: {trainAcc.toFixed(0)}%. Deeper trees carve finer step-like
        boundaries and grow more nodes — at some point they start fitting noise rather than signal.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'knn', label: 'K-Nearest Neighbors', sub: 'Classify by closest points' },
  { id: 'trees', label: 'Decision Trees', sub: 'Recursive rule-based splits' },
];

function SectionKNN() {
  return (
    <div>
      <P>
        K-Nearest Neighbors is an intuitive, non-parametric "lazy learner" — it stores the entire
        training set and, at prediction time, looks at the K closest points to decide a label.
        There's no real "training" step beyond storing the data.
      </P>

      <H2 c="Distance Metrics" />
      <Table
        heads={['Metric', 'Formula', 'Intuition']}
        rows={[
          ['Euclidean', '√Σ(xᵢ - yᵢ)²', 'Straight-line ("as the crow flies") distance — the default choice'],
          ['Manhattan', 'Σ|xᵢ - yᵢ|', '"City block" distance — sum of axis-aligned steps, less sensitive to diagonal outliers'],
          ['Minkowski', '(Σ|xᵢ - yᵢ|ᵖ)^(1/p)', 'Generalizes both — p=2 is Euclidean, p=1 is Manhattan'],
        ]}
      />

      <H2 c="Choosing K" />
      <P>
        K controls the bias-variance tradeoff directly. A small K (e.g. 1) follows the training data
        closely — low bias, high variance, prone to overfitting on noise. A large K averages over more
        neighbors — smoother, more biased, but far less sensitive to any single noisy point.
      </P>

      <H2 c="Try It — Slide K" />
      <P>Same 52 points (including a handful of deliberately mislabeled ones) — watch the boundary calm down as K grows.</P>
      <KNNDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# KNN is distance-based, so unscaled features with large ranges dominate the distance
X_scaled = StandardScaler().fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_scaled, y_train)`}</Code>

      <Note color="warning" icon="ti-alert-triangle">
        <strong>Curse of dimensionality:</strong> as the number of features grows, distances between
        points converge — everything ends up roughly equally "far," which erodes the whole notion of a
        "nearest" neighbor. KNN degrades noticeably past a few dozen dimensions unless paired with
        dimensionality reduction first.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why does feature scaling matter so much more for KNN than for, say, a decision tree?"
          a="KNN's predictions come directly from a distance calculation across all features at once, so a feature with a naturally larger numeric range (e.g. income in dollars vs. age in years) silently dominates the distance regardless of its actual predictive relevance. A decision tree splits one feature at a time on a threshold, so relative scale between different features never enters the comparison." />
      <QA q="What's the practical downside of choosing K=1?"
          a="The model has zero tolerance for noise — a single mislabeled or unusual training point creates its own little region of wrong predictions around itself. It's the textbook definition of high variance: the fitted boundary changes drastically with small changes to the training data." />
      <QA q="KNN is called a 'lazy' learning algorithm. What does that mean, and what's the tradeoff?"
          a="It does no real work at training time — it just stores the data. All the computation is deferred to prediction time, when it has to search the stored data for nearest neighbors. The tradeoff is near-instant 'training' but slow, memory-heavy prediction, especially as the dataset grows — the opposite profile of a model like logistic regression, which is slow to train but nearly free to predict with." />
    </div>
  );
}

function SectionTrees() {
  return (
    <div>
      <P>
        Decision Trees learn by recursively splitting the feature space into simpler, rule-based
        regions — each split asks a single yes/no question about one feature.
      </P>

      <H2 c="Splitting Criteria" />
      <Table
        heads={['Task', 'Criterion', 'What it measures']}
        rows={[
          ['Classification', 'Gini Impurity', 'Probability of misclassifying a randomly picked point if labeled by the node’s class mix'],
          ['Classification', 'Entropy / Information Gain', 'Reduction in label "surprise" (bits of information) after the split'],
          ['Regression', 'Variance Reduction (MSE)', 'How much a split reduces the spread of target values within each child'],
        ]}
      />
      <Mx block>{`  Gini impurity of a node with class-1 fraction p:

    Gini = 1 - p² - (1-p)²

  A split is chosen to minimize the size-weighted Gini of the two children —
  0 means a perfectly pure node (only one class present).`}</Mx>

      <H2 c="Structure" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="Root node">The entire training set, before any split has been made.</Card>
        <Card color="success" title="Internal nodes">A decision rule — "feature ≤ threshold?" — routing points left or right.</Card>
        <Card color="purple" title="Leaf nodes">A final prediction — the majority class (or mean, for regression) of whatever points land there.</Card>
      </Grid>

      <H2 c="Overfitting & Pruning" />
      <P>
        Left alone, a tree keeps splitting until every leaf is perfectly pure — memorizing the training
        set rather than learning a generalizable pattern. Controlling growth is essential:
      </P>
      <Table
        heads={['Control', 'What it limits']}
        rows={[
          ['max_depth', 'How many splits deep the tree is allowed to grow'],
          ['min_samples_split', 'The minimum number of samples a node must have before it’s allowed to split further'],
          ['Post-pruning (cost-complexity)', 'Grows the full tree first, then trims back branches that don’t improve validation performance enough to justify their complexity'],
        ]}
      />
      <Note color="info" icon="ti-arrow-right">
        Small K and deep, unpruned trees both overfit; large K and shallow trees both underfit — it's
        the same general bias-variance tradeoff either way. See "Validation & Bias-Variance" under
        Model Evaluation & Validation for the framework behind why.
      </Note>

      <H2 c="Try It — Grow the Tree" />
      <P>Same 56 points — slide max_depth and watch both the tree diagram and the decision boundary grow together.</P>
      <DecisionTreeDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
tree.fit(X_train, y_train)

print(tree.feature_importances_)   # which features drove the splits`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why is Gini impurity 0 for a pure node, and what's its maximum value for a binary split?"
          a="Gini = 1 - p² - (1-p)² is the probability of misclassifying a randomly-chosen point if you labeled it randomly according to the node's class proportions. If a node is pure (p=0 or p=1), that probability is exactly zero — there's no way to misclassify. For a binary class problem, Gini is maximized at p=0.5, where it equals 0.5 — the point of maximum uncertainty." />
      <QA q="A decision tree gets 99% train accuracy but only 70% test accuracy. What's the most likely fix?"
          a="This is classic overfitting — the tree has grown deep enough to memorize noise in the training set. The fix is to constrain growth: lower max_depth, raise min_samples_split/min_samples_leaf, or apply cost-complexity pruning after growing the full tree, then re-check on a validation set." />
      <QA q="Why don't decision trees need feature scaling the way KNN or SVMs do?"
          a="A tree's splits only ever compare one feature to a threshold at a time (feature ≤ t?) — the comparison is invariant to any monotonic rescaling of that feature, since it doesn't combine multiple features' magnitudes the way a distance calculation or a weighted linear sum does." />
    </div>
  );
}

const SECTION_MAP = {
  knn: <SectionKNN />,
  trees: <SectionTrees />,
};

export default function KNNDecisionTrees() {
  const [active, setActive] = useState('knn');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 05</div>
        <h1 className="page-header-title">KNN & Decision Trees</h1>
        <p className="page-header-subtitle">
          Two intuitive, non-linear ways to learn from data directly — classify by nearest neighbors,
          or split the feature space recursively into decision rules.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={5} />
    </div>
  );
}
