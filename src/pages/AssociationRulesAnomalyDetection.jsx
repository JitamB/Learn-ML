import { useState, useMemo } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
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

/* ── Association Rules demo: real Apriori mining + circular node-link graph ── */
const APRIORI_ITEMS = ['bread', 'milk', 'eggs', 'butter', 'diapers', 'beer', 'chips', 'soda', 'cheese', 'coffee', 'sugar', 'cereal'];
const APRIORI_PAIRS = [['chips', 'soda'], ['milk', 'cereal'], ['bread', 'butter'], ['coffee', 'sugar']];
const APRIORI_TRANSACTIONS = (() => {
  const rand = seededRandom(2457);
  const baseline = {};
  APRIORI_ITEMS.forEach(it => { baseline[it] = 0.13 + rand() * 0.05; });
  const N_TX = 60;
  const transactions = [];
  for (let t = 0; t < N_TX; t++) {
    const basket = new Set();
    const driverIncluded = {};
    APRIORI_PAIRS.forEach(([driver]) => {
      const inc = rand() < baseline[driver];
      driverIncluded[driver] = inc;
      if (inc) basket.add(driver);
    });
    APRIORI_PAIRS.forEach(([driver, target]) => {
      const p = driverIncluded[driver] ? 0.78 : baseline[target];
      if (rand() < p) basket.add(target);
    });
    APRIORI_ITEMS.forEach(it => {
      if (APRIORI_PAIRS.some(([d, tg]) => d === it || tg === it)) return;
      if (rand() < baseline[it]) basket.add(it);
    });
    transactions.push([...basket]);
  }
  return transactions;
})();
const APRIORI_PAIR_STATS = (() => {
  const N = APRIORI_TRANSACTIONS.length;
  const txSets = APRIORI_TRANSACTIONS.map(tx => new Set(tx));
  const supportOf = itemset => txSets.filter(tx => itemset.every(it => tx.has(it))).length / N;
  const singleSupport = {};
  APRIORI_ITEMS.forEach(it => { singleSupport[it] = supportOf([it]); });
  const pairs = [];
  for (let i = 0; i < APRIORI_ITEMS.length; i++) {
    for (let j = i + 1; j < APRIORI_ITEMS.length; j++) {
      const A = APRIORI_ITEMS[i], B = APRIORI_ITEMS[j];
      const suppAB = supportOf([A, B]);
      if (suppAB === 0 || !singleSupport[A] || !singleSupport[B]) continue;
      const confAB = suppAB / singleSupport[A], confBA = suppAB / singleSupport[B];
      const lift = confAB / singleSupport[B];
      const forward = confAB >= confBA ? { from: A, to: B, confidence: confAB } : { from: B, to: A, confidence: confBA };
      pairs.push({ A, B, support: suppAB, lift, best: forward });
    }
  }
  return pairs.sort((a, b) => b.lift - a.lift);
})();
function isPlantedPair(p) { return APRIORI_PAIRS.some(([a, b]) => (a === p.A && b === p.B) || (a === p.B && b === p.A)); }

const APRIORI_R = 115, APRIORI_CX = 150, APRIORI_CY = 150;
const APRIORI_POS = APRIORI_ITEMS.map((_, i) => {
  const angle = (i / APRIORI_ITEMS.length) * 2 * Math.PI - Math.PI / 2;
  return { x: APRIORI_CX + APRIORI_R * Math.cos(angle), y: APRIORI_CY + APRIORI_R * Math.sin(angle) };
});
const APRIORI_INDEX = Object.fromEntries(APRIORI_ITEMS.map((it, i) => [it, i]));

function AprioriDemo() {
  const [minSupport, setMinSupport] = useState(0.05);
  const [minConfidence, setMinConfidence] = useState(0.4);
  const surviving = useMemo(
    () => APRIORI_PAIR_STATS.filter(p => p.support >= minSupport && p.best.confidence >= minConfidence),
    [minSupport, minConfidence]
  );

  return (
    <VizBox>
      <SliderRow label="Minimum support" min={0.02} max={0.22} step={0.005} value={minSupport} onChange={setMinSupport} fmt={v => v.toFixed(3)} />
      <SliderRow label="Minimum confidence" min={0.3} max={0.9} step={0.02} value={minConfidence} onChange={setMinConfidence} fmt={v => v.toFixed(2)} />
      <Grid cols={2} gap={12}>
        <div>
          <svg viewBox="0 0 300 300" style={{ width: '100%', height: 'auto', display: 'block', border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)' }}>
            {surviving.map((p, i) => {
              const a = APRIORI_POS[APRIORI_INDEX[p.A]], b = APRIORI_POS[APRIORI_INDEX[p.B]];
              return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={isPlantedPair(p) ? 'var(--color-border-info)' : 'var(--color-border-secondary)'}
                strokeWidth={Math.min(4, 1 + p.lift / 1.5)} opacity={0.8} />;
            })}
            {APRIORI_ITEMS.map((it, i) => (
              <g key={it}>
                <circle cx={APRIORI_POS[i].x} cy={APRIORI_POS[i].y} r={4} fill="var(--color-text-secondary)" />
                <text x={APRIORI_POS[i].x} y={APRIORI_POS[i].y + (APRIORI_POS[i].y > APRIORI_CY ? 16 : -8)}
                  fontSize={10} textAnchor="middle" fill="var(--color-text-secondary)">{it}</text>
              </g>
            ))}
          </svg>
        </div>
        <div style={{ maxHeight: 260, overflowY: 'auto' }}>
          <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--color-text-tertiary)', textAlign: 'left' }}>
                <th style={{ padding: '2px 4px' }}>Rule</th><th>Supp.</th><th>Conf.</th><th>Lift</th>
              </tr>
            </thead>
            <tbody>
              {surviving.map((p, i) => (
                <tr key={i} style={{ borderTop: '1px solid var(--color-border-tertiary)' }}>
                  <td style={{ padding: '3px 4px', color: isPlantedPair(p) ? 'var(--color-text-info)' : 'var(--color-text-secondary)' }}>{p.best.from} → {p.best.to}</td>
                  <td>{p.support.toFixed(2)}</td>
                  <td>{p.best.confidence.toFixed(2)}</td>
                  <td>{p.lift.toFixed(2)}</td>
                </tr>
              ))}
              {surviving.length === 0 && <tr><td colSpan={4} style={{ padding: 6, color: 'var(--color-text-tertiary)' }}>No rules survive at these thresholds.</td></tr>}
            </tbody>
          </table>
        </div>
      </Grid>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        {surviving.length} rule{surviving.length === 1 ? '' : 's'} survive. Blue edges/rows are the
        4 pairs deliberately planted with real co-occurrence in this dataset — they should be the
        last ones standing as you raise both sliders.
      </div>
    </VizBox>
  );
}

/* ── Anomaly Detection demo: real, simplified Isolation Forest ── */
const ISO_POINTS = (() => {
  const randA = seededRandom(31), randB = seededRandom(67);
  const inliers = Array.from({ length: 50 }, () => ({ x: randNormalish(randA, 5, 1.0), y: randNormalish(randB, 5, 1.0) }));
  const outliers = [
    { x: 0.4, y: 0.5 }, { x: 9.6, y: 9.5 }, { x: 0.5, y: 9.3 },
    { x: 9.4, y: 0.6 }, { x: 0.3, y: 5.1 }, { x: 9.7, y: 4.8 },
  ];
  const borderline = [{ x: 7.6, y: 7.8 }, { x: 2.5, y: 7.5 }];
  return inliers.concat(outliers, borderline);
})();
function buildRandomTree(points, rand, depth, maxDepth) {
  if (depth >= maxDepth || points.length <= 1) return { isLeaf: true, size: points.length };
  const feat = rand() < 0.5 ? 'x' : 'y';
  const vals = points.map(p => p[feat]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) return { isLeaf: true, size: points.length };
  const threshold = lo + rand() * (hi - lo);
  const left = points.filter(p => p[feat] <= threshold);
  const right = points.filter(p => p[feat] > threshold);
  if (left.length === 0 || right.length === 0) return { isLeaf: true, size: points.length };
  return { isLeaf: false, feat, threshold, left: buildRandomTree(left, rand, depth + 1, maxDepth), right: buildRandomTree(right, rand, depth + 1, maxDepth) };
}
function cFactor(n) { return n <= 1 ? 0 : 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n); }
function pathLength(node, point, depth) {
  if (node.isLeaf) return depth + cFactor(node.size);
  const v = point[node.feat];
  return v <= node.threshold ? pathLength(node.left, point, depth + 1) : pathLength(node.right, point, depth + 1);
}
const ISO_TREES = (() => {
  const rand = seededRandom(999);
  return Array.from({ length: 50 }, () => buildRandomTree(ISO_POINTS, rand, 0, 8));
})();
const ISO_CN = cFactor(ISO_POINTS.length);
const ISO_SCORES = ISO_POINTS.map(p => {
  const avgPath = ISO_TREES.reduce((s, t) => s + pathLength(t, p, 0), 0) / ISO_TREES.length;
  return Math.pow(2, -avgPath / ISO_CN);
});

function IsolationForestDemo() {
  const [threshold, setThreshold] = useState(0.60);
  const flaggedCount = ISO_SCORES.filter(s => s >= threshold).length;

  return (
    <VizBox>
      <SliderRow label="Anomaly score threshold" min={0.35} max={0.75} step={0.01} value={threshold} onChange={setThreshold} fmt={v => v.toFixed(2)} />
      <div style={{ position: 'relative', height: 260, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden', background: 'var(--color-background-secondary)' }}>
        {ISO_POINTS.map((p, i) => {
          const flagged = ISO_SCORES[i] >= threshold;
          return (
            <div key={i} style={{
              position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
              transform: 'translate(-50%,-50%)', fontSize: 13, fontWeight: 700, lineHeight: 1,
              color: flagged ? 'var(--color-text-danger)' : 'var(--color-text-info)',
            }}>{flagged ? '✕' : '○'}</div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        {flaggedCount} of {ISO_POINTS.length} points flagged (✕) as anomalies at this threshold.{' '}
        {threshold < 0.45 ? 'Too loose — normal points near the edge of the main cloud are getting swept up too.'
          : threshold > 0.68 ? 'Too strict — some genuine outliers are slipping through unflagged.'
          : 'A reasonable middle ground: the 6 far-flung outliers get caught with little to no collateral damage on the main cloud.'}
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'rules', label: 'Association Rules', sub: 'Market-basket analysis' },
  { id: 'anomaly', label: 'Anomaly Detection', sub: 'Isolation Forest & One-Class SVM' },
];

function SectionRules() {
  const example = APRIORI_PAIR_STATS.find(p => (p.A === 'chips' && p.B === 'soda') || (p.A === 'soda' && p.B === 'chips'));
  return (
    <div>
      <P>
        Association rule mining answers a very literal question: reading through a pile of grocery
        receipts, which items keep showing up together more often than random chance would predict?
        Apriori and FP-Growth are two different algorithms for finding those patterns efficiently.
      </P>

      <H2 c="The Three Numbers That Matter" />
      <Table
        heads={['Metric', 'Formula', 'Question it answers']}
        rows={[
          ['Support', 'P(A and B)', 'How common is this combination overall?'],
          ['Confidence', 'P(B | A) = support(A,B) / support(A)', 'Given someone bought A, how often did they also buy B?'],
          ['Lift', 'confidence(A→B) / P(B)', 'Is this a real association, or just because B is popular anyway?'],
        ]}
      />
      <P>
        Lift is the one that actually separates signal from noise: a lift of exactly 1 means A and B
        are statistically independent — knowing about A told you nothing new about B. Lift well above
        1 means a genuine association; below 1 means buying A makes B <em>less</em> likely.
      </P>
      {example && (
        <>
          <P>Worked from this exact demo's data: <Mx>{example.A}</Mx> and <Mx>{example.B}</Mx> appear together in {(example.support * APRIORI_TRANSACTIONS.length).toFixed(0)} of {APRIORI_TRANSACTIONS.length} transactions.</P>
          <Mx block>{`  support(chips, soda) = ${(example.support * APRIORI_TRANSACTIONS.length).toFixed(0)} / ${APRIORI_TRANSACTIONS.length} = ${example.support.toFixed(3)}
  confidence(chips → soda) = ${example.best.confidence.toFixed(3)}   (of everyone who bought chips, this fraction also bought soda)
  lift = confidence / P(soda) = ${example.lift.toFixed(2)}   (${example.lift.toFixed(1)}× more likely together than chance alone would predict)`}</Mx>
        </>
      )}

      <H2 c="Apriori's Key Trick" />
      <P>
        Checking every possible combination of items is combinatorially hopeless even for a modest
        catalog. Apriori's shortcut is the observation that <strong>a frequent itemset's every
        subset must also be frequent</strong> — so it builds up candidates size by size (single
        items, then pairs, then triples...), pruning away anything whose subset already failed the
        minimum-support bar, before ever counting it.
      </P>
      <Note color="info" icon="ti-info-circle">
        FP-Growth solves the same problem without generating candidates at all — it compresses the
        whole transaction database into a tree structure (an "FP-tree") and mines frequent patterns
        directly from that, which is typically much faster on large datasets.
      </Note>

      <H2 c="Try It — Filter Rules by Support and Confidence" />
      <P>12 grocery items, 60 transactions, with 4 pairs deliberately given real co-occurrence. The graph draws an edge for every surviving rule; the table shows their exact numbers.</P>
      <AprioriDemo />

      <H3 c="A minimal working example" />
      <Code>{`from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(basket_df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does a lift value below 1 matter, and what does it actually mean?"
          a="Lift below 1 means the two items are negatively associated — buying A makes B less likely than it would be by chance, not just 'less strongly positive.' This is genuinely useful information (e.g. two competing products, or a health-food item negatively associated with snack purchases) that a support/confidence-only view would miss, since confidence alone can look deceptively high just because B is popular on its own." />
      <QA q="Why can a rule have high confidence but be practically useless?"
          a="Confidence only measures P(B|A), ignoring how common B already is on its own — if B is bought in 95% of all transactions regardless, then confidence(A→B) will look high for almost any A purely because B is nearly always present, not because A and B have any real relationship. Lift corrects for exactly this by dividing out B's baseline popularity." />
      <QA q="What is the 'apriori property' that makes the Apriori algorithm efficient, stated precisely?"
          a="If an itemset is frequent (meets the minimum support threshold), then every one of its subsets must also be frequent — equivalently, if any subset of an itemset is infrequent, the itemset itself cannot be frequent. This lets the algorithm prune away huge numbers of candidate itemsets at each size level without ever having to count their support directly." />
      <QA q="A retailer wants rules for a catalog of 100,000 SKUs. Would you reach for Apriori or FP-Growth, and why?"
          a="FP-Growth, because it avoids Apriori's expensive candidate-generation-and-counting step entirely by compressing the transaction database into a tree once and mining directly from that structure — Apriori's repeated database scans and combinatorial candidate generation become a serious bottleneck at that scale, while FP-Growth typically scales substantially better." />
    </div>
  );
}

function SectionAnomaly() {
  return (
    <div>
      <P>
        Anomaly detection flips the usual modeling question around: instead of learning what a
        pattern looks like, it learns what <em>normal</em> looks like well enough to notice when
        something clearly doesn't fit — with no labeled examples of "anomaly" to learn from at all.
      </P>

      <H2 c="Isolation Forest — The Core Insight" />
      <P>
        Build a tree by repeatedly picking a random feature and a random split point, and keep going
        until every point is alone in its own leaf. An outlier — sitting far from everything else —
        tends to get isolated in just a handful of lucky random splits. A normal point, buried deep
        inside a dense cluster, takes many more splits to separate from all its neighbors.
      </P>
      <Mx block>{`  anomaly_score(x) = 2^(-avg_path_length(x) / c(n))

  short average path  → isolated easily → HIGH anomaly score
  long average path    → deep in the crowd → LOW anomaly score

  c(n) is just a normalizing constant (the expected path length
  for a random point in a tree of n samples) so scores land in [0,1]
  regardless of dataset size.`}</Mx>
      <P>
        Averaging this path length across many independently-randomized trees (a "forest" of them,
        exactly like Random Forest's bagging idea, but with random rather than optimized splits)
        turns one lucky/unlucky tree into a stable, reliable score.
      </P>

      <H2 c="One-Class SVM" />
      <P>
        A different approach entirely: fit a boundary (in the SVM sense — potentially non-linear via
        a kernel) around the dense region containing "normal" data, then flag anything falling
        outside that boundary. It tends to work well when normal data forms one fairly coherent
        region, but scales less gracefully to very large datasets than Isolation Forest.
      </P>

      <H2 c="Try It — Tune the Anomaly Threshold" />
      <P>50 ordinary points, 6 planted far-away outliers, 2 ambiguous "borderline" points. The forest is built once — the slider only changes where the cutoff line falls on already-computed scores.</P>
      <IsolationForestDemo />
      <Note color="warning" icon="ti-alert-triangle">
        Real-world anomaly detection almost always trades off false positives against false
        negatives, same as any classifier — there is rarely a threshold with zero of both. Notice
        the borderline points in this demo: they get flagged before the threshold is strict enough
        to also risk missing a genuine outlier.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
labels = iso.fit_predict(X)      # -1 = anomaly, 1 = normal
scores = iso.decision_function(X)  # higher = more normal`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does Isolation Forest use random splits instead of the optimized (e.g. Gini-based) splits a decision tree normally uses?"
          a="The whole method relies on the fact that anomalies are, definitionally, easy to isolate almost regardless of which feature or threshold you happen to pick — optimized splits are built to separate classes as efficiently as possible, which would defeat the purpose here, since it would make isolating any point (anomalous or not) artificially fast and erase the very signal (short path length) the algorithm is measuring." />
      <QA q="What does scikit-learn's 'contamination' parameter actually control in Isolation Forest?"
          a="It sets the expected proportion of anomalies in the dataset, which the algorithm uses to choose where along the continuous anomaly-score scale to draw the flagged/not-flagged cutoff line — it doesn't change how the trees are built or how scores are computed, only where the final binary decision threshold lands." />
      <QA q="Why might Isolation Forest scale better to large datasets than a One-Class SVM?"
          a="Building a random tree over the data is roughly O(n log n), and prediction for a new point is roughly O(log n) per tree — both cheap and easily parallelizable across trees. A kernel-based One-Class SVM typically needs to compute or approximate relationships across many pairs of training points, which tends to scale considerably worse as the training set grows large." />
      <QA q="A dataset has no labeled anomalies at all. How would you even validate that an anomaly detector is working reasonably well?"
          a="Common approaches include manually inspecting the top-scored 'most anomalous' points for plausibility, deliberately injecting a small number of synthetic extreme outliers and checking the model reliably flags them, or — if any partial labels exist even informally — checking flagged points against them after the fact. In practice, anomaly detection is often validated more qualitatively than typical supervised metrics allow for." />
    </div>
  );
}

const SECTION_MAP = {
  rules: <SectionRules />,
  anomaly: <SectionAnomaly />,
};

export default function AssociationRulesAnomalyDetection() {
  const [active, setActive] = useState('rules');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 11</div>
        <h1 className="page-header-title">Association Rules & Anomaly Detection</h1>
        <p className="page-header-subtitle">
          Mine "if this, then that" patterns in transactional data, and flag points that don't fit
          any learned pattern at all.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={11} />
    </div>
  );
}
