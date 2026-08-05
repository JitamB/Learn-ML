import { useState, useMemo, useEffect } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
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
const sigmoid = z => 1 / (1 + Math.exp(-z));

/* ── Pseudo-Labeling demo: real logistic regression + confidence threshold ── */
const PL_POINTS = (() => {
  const rand = seededRandom(19);
  const pts = [];
  for (let i = 0; i < 20; i++) {
    const x = rand() * 10, y = rand() * 10;
    const noise = (rand() - 0.5) * 3.5;
    pts.push({ x, y, trueLabel: (x + y + noise > 10) ? 1 : 0 });
  }
  return pts;
})();
const PL_LABELED_IDX = (() => {
  const scored = PL_POINTS.map((p, idx) => ({ idx, dist: p.x + p.y - 10, trueLabel: p.trueLabel }));
  const pos = scored.filter(p => p.trueLabel === 1).sort((a, b) => b.dist - a.dist).slice(0, 3);
  const neg = scored.filter(p => p.trueLabel === 0).sort((a, b) => a.dist - b.dist).slice(0, 3);
  return new Set([...pos, ...neg].map(p => p.idx));
})();
const PL_FIT = (() => {
  const labeled = PL_POINTS.filter((_, i) => PL_LABELED_IDX.has(i));
  let w1 = 0, w2 = 0, b = 0;
  const lr = 0.01, l2 = 0.01, iters = 2000;
  for (let it = 0; it < iters; it++) {
    let gw1 = 0, gw2 = 0, gb = 0;
    labeled.forEach(p => {
      const err = sigmoid(w1 * p.x + w2 * p.y + b) - p.trueLabel;
      gw1 += err * p.x; gw2 += err * p.y; gb += err;
    });
    w1 -= lr * (gw1 / labeled.length + l2 * w1);
    w2 -= lr * (gw2 / labeled.length + l2 * w2);
    b -= lr * (gb / labeled.length);
  }
  return { w1, w2, b };
})();
const PL_LABELED = PL_POINTS.map((p, idx) => ({ ...p, idx })).filter(p => PL_LABELED_IDX.has(p.idx));
const PL_UNLABELED = PL_POINTS.map((p, idx) => ({ ...p, idx })).filter(p => !PL_LABELED_IDX.has(p.idx)).map(p => {
  const prob = sigmoid(PL_FIT.w1 * p.x + PL_FIT.w2 * p.y + PL_FIT.b);
  const predLabel = prob > 0.5 ? 1 : 0;
  return { ...p, prob, predLabel, confidence: Math.max(prob, 1 - prob), wrong: predLabel !== p.trueLabel };
});

function PseudoLabelDemo() {
  const [threshold, setThreshold] = useState(0.75);
  const selected = PL_UNLABELED.filter(p => p.confidence >= threshold);
  const wrongSelected = selected.filter(p => p.wrong);

  return (
    <VizBox>
      <SliderRow label="Confidence threshold" min={0.5} max={0.9} step={0.01} value={threshold} onChange={setThreshold} fmt={v => v.toFixed(2)} />
      <div style={{ position: 'relative', height: 240, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'visible' }}>
        {PL_LABELED.map(p => (
          <div key={p.idx} title={`labeled, class ${p.trueLabel}`} style={{
            position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
            transform: 'translate(-50%,-50%)', width: 15, height: 15, borderRadius: '50%',
            background: p.trueLabel ? 'var(--color-background-danger)' : 'var(--color-background-info)',
            border: `2.5px solid ${p.trueLabel ? 'var(--color-border-danger)' : 'var(--color-border-info)'}`,
          }} />
        ))}
        {PL_UNLABELED.map(p => {
          const isSelected = p.confidence >= threshold;
          return (
            <div key={p.idx} style={{
              position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
              transform: 'translate(-50%,-50%)', width: isSelected ? 11 : 7, height: isSelected ? 11 : 7, borderRadius: '50%',
              background: isSelected ? `var(--color-background-${p.predLabel ? 'danger' : 'info'})` : 'var(--color-border-tertiary)',
              border: isSelected ? `1.5px solid var(--color-border-${p.predLabel ? 'danger' : 'info'})` : 'none',
              opacity: isSelected ? 1 : 0.55,
            }}>
              {isSelected && p.wrong && (
                <i className="ti ti-alert-triangle" style={{ position: 'absolute', top: -15, left: -2, fontSize: 13, color: 'var(--color-text-warning)' }} />
              )}
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Bold ringed dots are the 6 truly labeled points. Small gray dots are unlabeled and below
        threshold. Past the threshold, an unlabeled point gets pseudo-labeled and colored by its{' '}
        <em>predicted</em> class — a ⚠ flags the (threshold-dependent) cases where that confident
        prediction is actually wrong, using ground truth a real pipeline wouldn't get to peek at.
        Selected: {selected.length}/14 · wrong among selected: {wrongSelected.length}.
      </div>
    </VizBox>
  );
}

/* ── Label Propagation demo: real k-NN graph + iterative harmonic propagation ── */
const LP_CENTERS = [[2.5, 2.5], [7.5, 2.8], [5.0, 7.8]];
const LP_TEXT = ['info', 'danger', 'success'];
const clampLP = v => Math.max(0.3, Math.min(9.7, v));
const LP_POINTS = (() => {
  const randX = seededRandom(77), randY = seededRandom(133);
  const points = [];
  LP_CENTERS.forEach(([cx, cy], blob) => {
    for (let i = 0; i < 8; i++) points.push({ x: clampLP(randNormalish(randX, cx, 1.3)), y: clampLP(randNormalish(randY, cy, 1.3)), blob });
  });
  return points;
})();
const LP_SEEDS = (() => {
  const seeds = {};
  LP_CENTERS.forEach(([cx, cy], blob) => {
    let best = -1, bestD = Infinity;
    LP_POINTS.forEach((p, idx) => { if (p.blob !== blob) return; const d = Math.hypot(p.x - cx, p.y - cy); if (d < bestD) { bestD = d; best = idx; } });
    seeds[best] = blob;
  });
  return seeds;
})();
const LP_EDGES = (() => {
  const k = 4, n = LP_POINTS.length, edgeMap = new Map();
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  for (let i = 0; i < n; i++) {
    const ds = [];
    for (let j = 0; j < n; j++) if (j !== i) ds.push({ j, d: dist(LP_POINTS[i], LP_POINTS[j]) });
    ds.sort((a, b) => a.d - b.d);
    for (let m = 0; m < k; m++) {
      const j = ds[m].j, key = i < j ? `${i}-${j}` : `${j}-${i}`;
      if (!edgeMap.has(key) || edgeMap.get(key) > ds[m].d) edgeMap.set(key, ds[m].d);
    }
  }
  return [...edgeMap.entries()].map(([key, w]) => { const [i, j] = key.split('-').map(Number); return { i, j, w }; });
})();
function lpPropagate(threshold, iters) {
  const n = LP_POINTS.length;
  const adj = Array.from({ length: n }, () => []);
  LP_EDGES.filter(e => e.w <= threshold).forEach(({ i, j }) => { adj[i].push(j); adj[j].push(i); });
  let F = Array.from({ length: n }, (_, i) => { const row = [0, 0, 0]; if (LP_SEEDS[i] !== undefined) row[LP_SEEDS[i]] = 1; return row; });
  for (let t = 0; t < iters; t++) {
    F = F.map((row, i) => {
      if (LP_SEEDS[i] !== undefined) return row;
      const nb = adj[i];
      if (nb.length === 0) return row;
      const sum = [0, 0, 0];
      nb.forEach(j => { for (let c = 0; c < 3; c++) sum[c] += F[j][c]; });
      return sum.map(v => v / nb.length);
    });
  }
  return F;
}
const LP_MAX_ITERS = 40;
function lpNodeStyle(i, F) {
  if (LP_SEEDS[i] !== undefined) {
    const c = LP_TEXT[LP_SEEDS[i]];
    return { r: 0.28, fill: `var(--color-background-${c})`, stroke: `var(--color-border-${c})`, strokeWidth: 0.06, opacity: 1 };
  }
  const row = F[i], s = row[0] + row[1] + row[2];
  if (s < 1e-6) return { r: 0.14, fill: 'var(--color-border-tertiary)', stroke: 'none', strokeWidth: 0, opacity: 0.7 };
  const maxVal = Math.max(...row), c = LP_TEXT[row.indexOf(maxVal)];
  return { r: 0.2, fill: `var(--color-background-${c})`, stroke: `var(--color-border-${c})`, strokeWidth: 0.04, opacity: 0.35 + (maxVal / s) * 0.65 };
}

function LabelPropagationDemo() {
  const [threshold, setThreshold] = useState(2.0);
  const [iter, setIter] = useState(LP_MAX_ITERS);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (iter >= LP_MAX_ITERS) { setPlaying(false); return; }
    const id = setTimeout(() => setIter(i => Math.min(LP_MAX_ITERS, i + 1)), 120);
    return () => clearTimeout(id);
  }, [playing, iter]);

  const F = useMemo(() => lpPropagate(threshold, iter), [threshold, iter]);
  const activeEdges = useMemo(() => LP_EDGES.filter(e => e.w <= threshold), [threshold]);
  const inactiveEdges = useMemo(() => LP_EDGES.filter(e => e.w > threshold), [threshold]);
  const unlabeledIdx = LP_POINTS.map((_, i) => i).filter(i => LP_SEEDS[i] === undefined);
  const reachedCount = unlabeledIdx.filter(i => Math.max(...F[i]) > 1e-6).length;

  return (
    <VizBox>
      <SliderRow label="Distance threshold" min={0.15} max={3.2} step={0.02} value={threshold} onChange={setThreshold} fmt={v => v.toFixed(2)} />
      <div style={{ display: 'flex', gap: 8, marginTop: 10, marginBottom: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <button onClick={() => setPlaying(p => !p)} style={toggleBtnStyle(playing)}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => setIter(i => Math.min(LP_MAX_ITERS, i + 1))} style={toggleBtnStyle(false)}>Step</button>
        <button onClick={() => { setIter(0); setPlaying(false); }} style={toggleBtnStyle(false)}>Reset</button>
        <span style={{ fontSize: 12, color: 'var(--color-text-tertiary)' }}>Iteration {iter} / {LP_MAX_ITERS}</span>
      </div>
      <svg viewBox="0 0 10 10" style={{ width: '100%', height: 280, background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
        {inactiveEdges.map((e, idx) => (
          <line key={'i' + idx} x1={LP_POINTS[e.i].x} y1={10 - LP_POINTS[e.i].y} x2={LP_POINTS[e.j].x} y2={10 - LP_POINTS[e.j].y}
            stroke="var(--color-border-tertiary)" strokeWidth={0.02} strokeDasharray="0.1,0.1" opacity={0.35} />
        ))}
        {activeEdges.map((e, idx) => (
          <line key={'a' + idx} x1={LP_POINTS[e.i].x} y1={10 - LP_POINTS[e.i].y} x2={LP_POINTS[e.j].x} y2={10 - LP_POINTS[e.j].y}
            stroke="var(--color-border-secondary)" strokeWidth={0.035} opacity={0.75} />
        ))}
        {LP_POINTS.map((p, i) => {
          const st = lpNodeStyle(i, F);
          return <circle key={i} cx={p.x} cy={10 - p.y} r={st.r} fill={st.fill} stroke={st.stroke} strokeWidth={st.strokeWidth} opacity={st.opacity} />;
        })}
      </svg>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Bold ringed dots are the 3 labeled seeds. {reachedCount}/{unlabeledIdx.length} unlabeled points
        have been reached so far — the rest stay gray, cut off from every seed by the current distance
        threshold. Fainter fill means a less confident soft label. Solid lines are edges the current
        threshold allows a label to cross; faint dashed lines are cut.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'pseudolabel', label: 'Pseudo-Labeling & Self-Training', sub: "Bootstrapping labels from a model's own confidence" },
  { id: 'labelprop', label: 'Label Propagation', sub: 'Letting labels flow through a similarity graph' },
];

function SectionPseudoLabel() {
  return (
    <div>
      <P>
        Semi-supervised learning sits between the two extremes: you have a small pool of hard-won
        labeled data and a much larger pile of unlabeled data sitting right next to it, and throwing
        that unlabeled pile away would waste most of what you have. It's like grading a stack of 100
        exams when you've only got an answer key for 10 of them — you'd use those 10 to build your
        best guess at a rubric, apply it to the other 90, and trust the results you're most confident
        about while double-checking the rest.
      </P>

      <H2 c="Pseudo-Labeling" />
      <P>
        Train an ordinary supervised model on the small labeled set, then use it to predict on the
        unlabeled set. For a classifier, every prediction comes with a confidence — for binary
        classification, <Mx>max(p, 1−p)</Mx> where <Mx>p</Mx> is the predicted probability of the
        positive class. Keep only the predictions above some confidence threshold, treat them as if
        they were real labels ("pseudo-labels"), fold them into the training set, and retrain.
      </P>
      <Mx block>{`  confidence(x) = max( P(y=1 | x),  1 − P(y=1 | x) )

  if confidence(x) ≥ threshold:  add (x, ŷ) to the labeled set as if ŷ were ground truth`}</Mx>

      <H2 c="Self-Training" />
      <P>
        Self-training is simply this process run to convergence instead of once: retrain on the
        expanded set, re-predict on whatever's still unlabeled, re-select the newly confident ones,
        fold them in, and repeat — until no more predictions clear the confidence bar, or the labeled
        set stops growing. Pseudo-labeling is the single-round special case of self-training's loop.
      </P>

      <H2 c="Try It — Move the Confidence Bar" />
      <P>
        6 labeled points (3 per class, the clearest examples of each) train a small logistic
        regression. The other 14 points are unlabeled; the slider controls how confident the model
        has to be before a prediction gets pseudo-labeled.
      </P>
      <PseudoLabelDemo />

      <H3 c="A worked example, using this exact fit" />
      <P>
        The trained weights are <Mx>w₁≈0.279, w₂≈0.324, b≈−2.680</Mx>. Take the unlabeled point at{' '}
        <Mx>(8.24, 7.98)</Mx>:
      </P>
      <Mx block>{`  z = 0.279×8.24 + 0.324×7.98 − 2.680 ≈ 2.21
  P(y=1) = sigmoid(2.21) ≈ 0.90

  → 90% confident, class 1 — comfortably past any reasonable threshold.`}</Mx>
      <P>
        Now compare the point at <Mx>(6.50, 4.63)</Mx>, whose true (hidden) class is actually 0:
      </P>
      <Mx block>{`  z = 0.279×6.50 + 0.324×4.63 − 2.680 ≈ 0.63
  P(y=1) = sigmoid(0.63) ≈ 0.65

  → 65% confident, class 1 — and wrong. At any threshold at or below ~0.65, this
    mistake gets pseudo-labeled and fed into the next training round as if it
    were correct.`}</Mx>
      <P>
        That second point is exactly the one flagged with a ⚠ in the demo above once the slider drops
        to around 0.65 or below — drag it down and watch it appear.
      </P>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

# y_unlabeled entries are -1 by sklearn's convention
base = SVC(probability=True, gamma="auto")
model = SelfTrainingClassifier(base, threshold=0.75)
model.fit(X, y)   # y mixes real labels and -1 for unlabeled rows`}</Code>

      <Note color="warning" icon="ti-alert-triangle">
        <strong>Confirmation bias is the central risk.</strong> A confident wrong prediction becomes
        "ground truth" for the next round, and the model can end up more confident in its own mistake
        each time it retrains — there's no external check to catch it. In practice this is managed
        with a high initial threshold, a capped number of rounds, and sometimes an ensemble voting on
        which pseudo-labels to trust.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why does pseudo-labeling use a confidence threshold instead of just labeling every unlabeled point?"
          a="Every prediction the base model makes is a guess, and low-confidence guesses are the ones most likely to be wrong. Folding a wrong pseudo-label into the training set as if it were real teaches the model to trust its own mistake — the confidence threshold is a (imperfect) filter meant to keep only the pseudo-labels the model is least likely to be wrong about." />
      <QA q="How is Self-Training different from Pseudo-Labeling, precisely?"
          a="Pseudo-labeling is usually described as a single round: predict once, keep the confident ones, retrain once. Self-training is the general iterative wrapper — retrain, re-predict on what's still unlabeled, re-select, and repeat until no more confident predictions remain to add. Pseudo-labeling is the one-round special case." />
      <QA q="What's confirmation bias in this context, and why is it hard to detect from inside the pipeline?"
          a="It's the failure mode where a confidently wrong prediction gets treated as ground truth, and each retraining round can make the model even more confident in that same mistake, since nothing in the pipeline ever checks the pseudo-label against reality. It's hard to detect internally because the model's own confidence score is exactly the signal that failed — there's no second, independent source of truth being consulted." />
      <QA q="Why does this demo train its base model on only 3 points per class instead of using all 6 labeled points evenly, or more?"
          a="It doesn't withhold any labeled data — all 6 labeled points are used to fit the model. The 3-per-class split just describes how those 6 were chosen from the full 20-point dataset in the first place (the clearest, least ambiguous example of each class), which is what makes the initial fit reasonably reliable despite having so little labeled data to start from." />
    </div>
  );
}

function SectionLabelPropagation() {
  return (
    <div>
      <P>
        Label propagation takes a completely different approach: instead of training a parametric
        model at all, it builds a graph connecting every point (labeled and unlabeled alike) to its
        nearest neighbors, and lets labels spread along that graph like dye dropped into water,
        flowing fastest through the pipes — edges — that connect similar points most directly.
      </P>

      <H2 c="The Smoothness Assumption" />
      <P>
        The whole method rests on one assumption: points that are close together (or reachable
        through a chain of close neighbors) probably share a label. If that's true of your data, a
        handful of labeled points can flood an entire well-connected cluster with the right answer —
        no model parameters, no training loop, just repeated averaging over the graph.
      </P>
      <Mx block>{`  F ← D⁻¹ W F        (every node's label distribution becomes the average
                       of its neighbors' current label distributions)

  then re-clamp every labeled row back to its one-hot true label, and repeat.

  W = the graph's edge weights (here: 1 if an edge is "active", 0 otherwise)
  D = the diagonal degree matrix (Dᵢᵢ = number of active edges at node i)
  F = an N × (number of classes) matrix; unlabeled rows start at all-zero`}</Mx>

      <H3 c="A tiny worked example" />
      <P>
        Four nodes in a line, <Mx>A – B – C – D</Mx>, where <Mx>A</Mx> is labeled class 1 and{' '}
        <Mx>D</Mx> is labeled class 2, and every edge is active. Starting from{' '}
        <Mx>F_A=[1,0], F_B=[0,0], F_C=[0,0], F_D=[0,1]</Mx>, one synchronous round averages each
        unlabeled node's neighbors' <em>current</em> values:
      </P>
      <Mx block>{`  F_B ← average(F_A, F_C) = average([1,0], [0,0]) = [0.5, 0]
  F_C ← average(F_B, F_D) = average([0,0], [0,1]) = [0, 0.5]

  then re-clamp: F_A = [1,0] (unchanged), F_D = [0,1] (unchanged)`}</Mx>
      <P>
        After just one round, <Mx>B</Mx> already leans toward <Mx>A</Mx>'s class and <Mx>C</Mx> leans
        toward <Mx>D</Mx>'s — a few more rounds keep sharpening that lean as the influence keeps
        flowing back and forth.
      </P>

      <H2 c="Try It — Grow the Threshold, Watch It Spread" />
      <P>
        24 points in 3 loose groups, each connected to its 4 nearest neighbors. Only 1 point per
        group starts labeled (bold ring); the other 21 start gray. Dragging the distance threshold
        shows the fully-propagated result at that threshold right away; hit Reset, then Play or Step,
        to watch propagation actually unfold from scratch against the current threshold instead.
      </P>
      <LabelPropagationDemo />
      <P s={{ marginTop: 10 }}>
        At a low threshold almost every edge is cut, so propagation can't reach anywhere — everything
        stays gray. Raise it past roughly 2.0 and all 21 points get reached, correctly, in just a
        handful of iterations. Push it further still and the graph starts reconnecting across group
        boundaries — watch the one point sitting right on a boundary start to waver.
      </P>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.semi_supervised import LabelPropagation

# y_unlabeled entries are -1 by sklearn's convention
model = LabelPropagation(kernel="knn", n_neighbors=4)
model.fit(X, y)
model.transduction_   # the fully-propagated labels for every point, including
                      # the ones that started at -1`}</Code>

      <Note color="info" icon="ti-info-circle">
        Propagation is only as trustworthy as the graph it runs on. A distance threshold set too high
        connects points across a real boundary, and a wrong label crosses that bridge with exactly
        the same confidence as a right one would — the algorithm has no way to tell the difference
        between a well-formed neighborhood and an accidental shortcut.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="What is the 'smoothness assumption' in label propagation, and what happens when it doesn't hold?"
          a="It's the assumption that points close together in the graph (directly or via a chain of neighbors) share the same label. If the true classes are actually intermingled at the scale the graph connects things — no clean neighborhoods to exploit — propagation has nothing reliable to spread, and can confidently produce the wrong answer just as easily as the right one." />
      <QA q="Why does label propagation re-clamp the labeled rows after every iteration instead of letting them update like everything else?"
          a="The labeled points are the only actual ground truth in the whole system — if they were allowed to drift during averaging, the few real anchors the algorithm has would dilute away and the whole graph could converge to a meaningless, uniform average instead of propagating real information outward from trustworthy sources." />
      <QA q="How is choosing the similarity graph's connectivity (here, a distance threshold) similar to choosing k in k-NN or epsilon in DBSCAN?"
          a="All three control how much of the data's local structure a single hyperparameter is allowed to bridge — too tight and real neighborhoods fragment (or, in DBSCAN's case, everything looks like noise); too loose and unrelated points get connected as if they were similar. It's the same bias-variance-flavored tradeoff wearing a different hyperparameter's name in each algorithm." />
      <QA q="Could label propagation work on data where nothing resembling Euclidean distance makes sense, like text or graph-structured data?"
          a="Yes — the only real requirement is some way to define a similarity or distance between any two points, which for text could be embedding-vector cosine similarity, and for graph-structured data could just be the existing graph edges themselves. The propagation update rule itself doesn't care what the underlying similarity measure was, only that a weighted graph exists to average over." />
    </div>
  );
}

const SECTION_MAP = {
  pseudolabel: <SectionPseudoLabel />,
  labelprop: <SectionLabelPropagation />,
};

export default function SemiSupervisedLearning() {
  const [active, setActive] = useState('pseudolabel');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 13</div>
        <h1 className="page-header-title">Semi-Supervised Learning</h1>
        <p className="page-header-subtitle">
          Learning from a small pool of labeled data plus a much larger pool of unlabeled data, for
          when labeling everything by hand isn't realistic.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={13} />
    </div>
  );
}
