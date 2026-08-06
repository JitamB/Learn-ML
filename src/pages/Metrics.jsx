import { useState, useMemo } from 'react';
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
const sigmoid = z => 1 / (1 + Math.exp(-z));

/* ── Demo 1: Confusion Matrix + Live ROC Curve ──────────────────
   Exact same dataset/weights as LinearModels.jsx's LogisticRegressionDemo
   — a deliberate "same dataset as Linear Models" callback. */
const CM_W1 = 1, CM_W2 = 1, CM_B = -10;
const CM_POINTS = (() => {
  const rand = seededRandom(7);
  return Array.from({ length: 40 }, () => {
    const x = rand() * 10, y = rand() * 10;
    const noise = (rand() - 0.5) * 4;
    return { x, y, label: (x + y + noise > 10) ? 1 : 0 };
  });
})();
function confusionAt(threshold) {
  let TP = 0, FP = 0, FN = 0, TN = 0;
  CM_POINTS.forEach(p => {
    const prob = sigmoid(CM_W1 * p.x + CM_W2 * p.y + CM_B);
    const pred = prob > threshold ? 1 : 0;
    if (pred === 1 && p.label === 1) TP++;
    else if (pred === 1 && p.label === 0) FP++;
    else if (pred === 0 && p.label === 1) FN++;
    else TN++;
  });
  return { TP, FP, FN, TN };
}
const ROC_CURVE = (() => {
  const pts = [];
  for (let t = 0; t <= 1.001; t += 0.004) {
    const { TP, FP, FN, TN } = confusionAt(Math.min(t, 1));
    pts.push({ fpr: FP / (FP + TN) || 0, tpr: TP / (TP + FN) || 0 });
  }
  return pts.sort((a, b) => a.fpr - b.fpr || a.tpr - b.tpr);
})();
function trapAUC(pts, xk, yk) {
  let auc = 0;
  for (let i = 1; i < pts.length; i++) auc += (pts[i][xk] - pts[i - 1][xk]) * (pts[i][yk] + pts[i - 1][yk]) / 2;
  return auc;
}
const ROC_AUC = trapAUC(ROC_CURVE, 'fpr', 'tpr');
const PR_CURVE = (() => {
  const pts = [];
  for (let t = 0; t <= 1.001; t += 0.004) {
    const { TP, FP, FN } = confusionAt(Math.min(t, 1));
    const recall = TP / (TP + FN) || 0;
    const precision = (TP + FP) > 0 ? TP / (TP + FP) : 1;
    pts.push({ recall, precision });
  }
  return pts.sort((a, b) => a.recall - b.recall || a.precision - b.precision);
})();
const PR_AUC = trapAUC(PR_CURVE, 'recall', 'precision');
const CM_GRID_COLS = 20, CM_GRID_ROWS = 14;

function ConfusionMatrixROCDemo() {
  const [threshold, setThreshold] = useState(0.5);
  const { TP, FP, FN, TN } = useMemo(() => confusionAt(threshold), [threshold]);
  const precision = TP / (TP + FP) || 0;
  const recall = TP / (TP + FN) || 0;
  const f1 = (precision + recall) ? 2 * precision * recall / (precision + recall) : 0;
  const fpr = FP / (FP + TN) || 0, tpr = TP / (TP + FN) || 0;

  const cells = useMemo(() => {
    const out = [];
    for (let r = 0; r < CM_GRID_ROWS; r++) {
      for (let c = 0; c < CM_GRID_COLS; c++) {
        const px = (c + 0.5) / CM_GRID_COLS * 10, py = 10 - (r + 0.5) / CM_GRID_ROWS * 10;
        out.push(sigmoid(CM_W1 * px + CM_W2 * py + CM_B) > threshold ? 1 : 0);
      }
    }
    return out;
  }, [threshold]);

  const rocW = 190, rocH = 190;
  const rocPath = 'M ' + ROC_CURVE.map(p => `${(p.fpr * rocW).toFixed(1)},${(rocH - p.tpr * rocH).toFixed(1)}`).join(' L ');

  return (
    <VizBox>
      <SliderRow label="Threshold" min={0.01} max={0.99} step={0.01} value={threshold} onChange={setThreshold} fmt={v => v.toFixed(2)} />
      <Grid cols={2} gap={18}>
        <div>
          <div style={{ position: 'relative', height: 200, marginTop: 8, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: `repeat(${CM_GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${CM_GRID_ROWS}, 1fr)` }}>
              {cells.map((cls, i) => <div key={i} style={{ background: cls ? 'var(--color-background-danger)' : 'var(--color-background-info)' }} />)}
            </div>
            {CM_POINTS.map((p, i) => (
              <div key={i} style={{
                position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
                transform: 'translate(-50%,-50%)', fontSize: 12, fontWeight: 700, lineHeight: 1,
                color: p.label ? 'var(--color-text-danger)' : 'var(--color-text-info)',
              }}>
                {p.label ? '✕' : '○'}
              </div>
            ))}
          </div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 6 }}>Shaded region = predicted class 1 at this threshold. ✕ = actual 1, ○ = actual 0.</div>
        </div>
        <div>
          <table style={{ borderCollapse: 'collapse', fontSize: 12.5, marginBottom: 12 }}>
            <tbody>
              <tr>
                <td />
                <td style={{ padding: '3px 8px', textAlign: 'center', color: 'var(--color-text-tertiary)', fontSize: 11 }}>pred 1</td>
                <td style={{ padding: '3px 8px', textAlign: 'center', color: 'var(--color-text-tertiary)', fontSize: 11 }}>pred 0</td>
              </tr>
              <tr>
                <td style={{ padding: '3px 8px', color: 'var(--color-text-tertiary)', fontSize: 11 }}>actual 1</td>
                <td style={{ padding: '10px 14px', textAlign: 'center', background: 'var(--color-background-success)', fontWeight: 700, borderRadius: 'var(--border-radius-sm)' }}>{TP}</td>
                <td style={{ padding: '10px 14px', textAlign: 'center', background: 'var(--color-background-danger)', fontWeight: 700, borderRadius: 'var(--border-radius-sm)' }}>{FN}</td>
              </tr>
              <tr>
                <td style={{ padding: '3px 8px', color: 'var(--color-text-tertiary)', fontSize: 11 }}>actual 0</td>
                <td style={{ padding: '10px 14px', textAlign: 'center', background: 'var(--color-background-danger)', fontWeight: 700, borderRadius: 'var(--border-radius-sm)' }}>{FP}</td>
                <td style={{ padding: '10px 14px', textAlign: 'center', background: 'var(--color-background-success)', fontWeight: 700, borderRadius: 'var(--border-radius-sm)' }}>{TN}</td>
              </tr>
            </tbody>
          </table>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
            <Badge color="info">P = {precision.toFixed(2)}</Badge>
            <Badge color="purple">R = {recall.toFixed(2)}</Badge>
            <Badge color="success">F1 = {f1.toFixed(2)}</Badge>
          </div>
          <svg viewBox={`0 0 ${rocW} ${rocH}`} width={rocW} height={rocH} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
            <line x1={0} y1={rocH} x2={rocW} y2={0} stroke="var(--color-border-tertiary)" strokeWidth={1} strokeDasharray="3,3" />
            <path d={rocPath} fill="none" stroke="var(--accent)" strokeWidth={2} />
            <circle cx={fpr * rocW} cy={rocH - tpr * rocH} r={5} fill="var(--color-background-danger)" stroke="var(--color-border-danger)" strokeWidth={2} />
          </svg>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginTop: 4 }}>ROC curve (AUC = {ROC_AUC.toFixed(3)}) — dot tracks the current threshold's (FPR, TPR).</div>
        </div>
      </Grid>
    </VizBox>
  );
}

/* ── Demo 2: Outlier sensitivity, MAE vs. RMSE ─────────────────
   Pure arithmetic, no fitting — 9 fixed "normal" residuals plus one
   adjustable-magnitude outlier. */
const OUTLIER_BASE = (() => {
  const rand = seededRandom(303);
  return Array.from({ length: 9 }, () => randNormalish(rand, 0, 0.6));
})();
function OutlierSensitivityDemo() {
  const [magnitude, setMagnitude] = useState(0);
  const residuals = [...OUTLIER_BASE, magnitude];
  const mae = residuals.reduce((s, r) => s + Math.abs(r), 0) / residuals.length;
  const rmse = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / residuals.length);
  const maxBar = 7;

  return (
    <VizBox>
      <SliderRow label="Outlier magnitude" min={0} max={20} step={0.5} value={magnitude} onChange={setMagnitude} fmt={v => v.toFixed(1)} />
      <div style={{ display: 'flex', gap: 28, marginTop: 16, alignItems: 'flex-end', height: 110 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ width: 54, height: Math.max(2, Math.min(100, mae / maxBar * 100)), background: 'var(--color-background-info)', borderRadius: '4px 4px 0 0', margin: '0 auto' }} />
          <div style={{ fontSize: 12.5, marginTop: 6, fontFamily: 'var(--font-mono)' }}>MAE = {mae.toFixed(2)}</div>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ width: 54, height: Math.max(2, Math.min(100, rmse / maxBar * 100)), background: 'var(--color-background-danger)', borderRadius: '4px 4px 0 0', margin: '0 auto' }} />
          <div style={{ fontSize: 12.5, marginTop: 6, fontFamily: 'var(--font-mono)' }}>RMSE = {rmse.toFixed(2)}</div>
        </div>
        <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: 'var(--color-text-secondary)', paddingBottom: 8 }}>
          RMSE / MAE = {(rmse / mae).toFixed(2)}
        </div>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        9 fixed, ordinary-sized residuals plus one adjustable outlier. MAE grows exactly linearly with
        the outlier's size (it's just an average); RMSE grows much faster, because squaring an error
        disproportionately punishes large ones — the ratio climbs from 1.30 at magnitude 0 to over
        2.6 by magnitude 20.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'classification', label: 'Classification Metrics', sub: 'Confusion matrix, precision/recall, ROC-AUC & beyond' },
  { id: 'regression', label: 'Regression Metrics', sub: 'MAE, RMSE, R² & knowing which to trust' },
  { id: 'unsupervised', label: 'Unsupervised Metrics', sub: 'Scoring a model with no ground truth to check against' },
];

function SectionClassification() {
  return (
    <div>
      <P>
        A classifier's raw predictions are just labels — turning them into a trustworthy verdict on
        "is this model good" requires picking the right lens, because different metrics disagree with
        each other constantly, and each one is blind to a different kind of mistake.
      </P>

      <H2 c="The Confusion Matrix" />
      <P>
        Every binary classification metric on this page is computed from just four counts, so it's
        worth internalizing this table before anything else:
      </P>
      <Table
        heads={['', 'Predicted Positive', 'Predicted Negative']}
        rows={[
          ['Actual Positive', 'True Positive (TP)', 'False Negative (FN) — Type II error'],
          ['Actual Negative', 'False Positive (FP) — Type I error', 'True Negative (TN)'],
        ]}
      />

      <H2 c="Precision & Recall" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Precision = TP / (TP + FP)">Of everything the model called positive, how much actually was? High cost of a false positive (spam filter deleting a real email) → optimize this.</Card>
        <Card color="purple" title="Recall = TP / (TP + FN)">Of everything actually positive, how much did the model find? High cost of a false negative (missing a cancer case) → optimize this.</Card>
      </Grid>

      <H2 c="F1 and the F_β Family" />
      <Mx block>{`  F1  = 2 · (Precision · Recall) / (Precision + Recall)     (harmonic mean — punishes
                                                             a big gap between P and R
                                                             more than a plain average would)

  F_β = (1 + β²) · (P · R) / (β² · P + R)

  β > 1  →  weights Recall more (e.g. F2 for disease screening: missing a
             real case is worse than one extra false alarm)
  β < 1  →  weights Precision more (e.g. F0.5 when false positives are
             expensive to act on)`}</Mx>

      <H2 c="The Accuracy Paradox" />
      <P>
        As mentioned when Data Cleaning & Feature Engineering covered imbalanced classes: on a
        dataset that's 99% negative, a classifier that unconditionally predicts "negative" scores 99%
        accuracy while catching precisely zero positives. This is exactly why precision, recall, and
        F1 — which are all sensitive to how the minority class is actually handled — exist as
        alternatives to plain accuracy.
      </P>

      <H2 c="ROC-AUC vs. PR-AUC" />
      <P>
        The <strong>ROC curve</strong> plots True Positive Rate (recall) against False Positive Rate
        at every possible threshold; the <strong>Precision-Recall curve</strong> plots Precision
        against Recall instead. Both curves' areas (AUC) are threshold-independent single-number
        summaries — but they answer subtly different questions, and can disagree.
      </P>
      <Note color="info" icon="ti-info-circle">
        On the exact 40-point dataset used in the demo below (15 positive / 25 negative — a mild,
        realistic imbalance), <strong>ROC-AUC = {ROC_AUC.toFixed(3)}</strong> while{' '}
        <strong>PR-AUC = {PR_AUC.toFixed(3)}</strong> — noticeably lower. ROC-AUC stays roughly stable even as the
        positive class gets rarer, because the False Positive Rate is measured against the (large)
        negative class; PR-AUC is directly sensitive to how rare the positive class is, which is
        exactly why PR-AUC is the sharper, more honest choice once the positive class is genuinely
        rare (fraud detection, disease screening) — ROC-AUC can look deceptively good on a problem
        where precision is actually terrible.
      </Note>

      <H2 c="Log Loss — A Proper Scoring Rule" />
      <Mx block>{`  Log Loss = −(1/n) Σᵢ [ yᵢ·log(p̂ᵢ) + (1−yᵢ)·log(1−p̂ᵢ) ]`}</Mx>
      <P>
        Unlike accuracy, F1, or even ROC-AUC — all of which only look at whether a hard label was
        right — log loss is a <strong>proper scoring rule</strong>: it's minimized exactly when the
        predicted probabilities match the true probabilities, so it rewards a model that says "73%
        confident" when it's genuinely right 73% of the time, and punishes overconfidence severely (a
        confidently wrong prediction, e.g. predicting 0.99 for a case that turns out negative, incurs
        a very large penalty). This matters whenever the probabilities themselves get used downstream
        (ranking, risk scores, decision thresholds chosen later) rather than just the final label.
      </P>

      <H2 c="Matthews Correlation Coefficient & Cohen's Kappa" />
      <P>
        Two single-number metrics that use <em>all four</em> confusion-matrix cells at once, both
        increasingly favored over plain accuracy or even F1 specifically because they stay meaningful
        under class imbalance.
      </P>
      <Mx block>{`  MCC = (TP·TN − FP·FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]     range: −1 to +1
        (+1 = perfect, 0 = no better than random, −1 = perfectly wrong)

  Cohen's Kappa = (Pₒ − Pₑ) / (1 − Pₑ)      Pₒ = observed agreement (= accuracy)
                                             Pₑ = agreement expected purely by chance,
                                                  given each class's marginal frequency`}</Mx>
      <P>
        On this page's own 40-point dataset at threshold 0.5 (<Mx>TP=13, FP=4, FN=2, TN=21</Mx>):{' '}
        <strong>MCC = 0.692</strong>, and <strong>Cohen's Kappa = 0.688</strong> (observed agreement
        <Mx>Pₒ = 0.850</Mx> versus a chance-expected agreement of only <Mx>Pₑ = 0.519</Mx> — the gap
        between the two is exactly what Kappa is crediting the model for).
      </P>

      <H2 c="Multi-Class Averaging: Macro, Micro & Weighted" />
      <P>
        Precision/Recall/F1 are inherently binary — for <Mx>k</Mx> classes, compute one F1 per class
        (treating it as "this class vs. everything else"), then combine:
      </P>
      <Table
        heads={['Averaging', 'How it combines per-class scores', 'Sensitive to...']}
        rows={[
          ['Macro', 'Plain average across classes', 'Small classes equally — one badly-served small class hurts the score just as much as a large one'],
          ['Weighted', 'Average weighted by each class\'s support (count)', 'Large classes more — dominated by whichever class has the most examples'],
          ['Micro', 'Pool all TP/FP/FN globally first, then compute one F1', 'The overall pool of correct vs. incorrect predictions'],
        ]}
      />
      <P>
        A worked example makes the difference concrete — a 3-class classifier with support 10/10/5:
      </P>
      <Table
        heads={['Class', 'Precision', 'Recall', 'F1', 'Support']}
        rows={[
          ['A', '0.889', '0.800', '0.842', '10'],
          ['B', '0.667', '0.600', '0.632', '10'],
          ['C', '0.571', '0.800', '0.667', '5'],
        ]}
      />
      <P>
        <strong>Macro F1 = 0.713</strong>, <strong>Weighted F1 = 0.723</strong>,{' '}
        <strong>Micro F1 = 0.720</strong>. Micro-F1 here is exactly equal to overall accuracy
        (18 correct / 25 total) — a real, general fact worth knowing: <em>in single-label multi-class
        classification, micro-F1 always equals accuracy.</em> The three numbers are close, but not
        identical — Macro treats class C's decent-but-imperfect performance as equally important as A
        and B despite having half their support; Weighted and Micro both lean toward the larger
        classes' performance instead.
      </P>

      <H2 c="Try It — Sweep the Threshold, Watch Every Metric Move Together" />
      <P>Same 40 labeled points and fitted model as Linear Models' Logistic Regression demo — only the threshold changes.</P>
      <ConfusionMatrixROCDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.metrics import (roc_auc_score, average_precision_score,
                              matthews_corrcoef, cohen_kappa_score, f1_score)

probs = clf.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)

print("ROC-AUC:", roc_auc_score(y_test, probs))
print("PR-AUC:", average_precision_score(y_test, probs))
print("MCC:", matthews_corrcoef(y_test, preds))
print("Macro F1:", f1_score(y_test, preds, average="macro"))`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why is accuracy a misleading primary metric for a dataset that's 99% one class?"
          a="A classifier that always predicts the majority class scores 99% accuracy while never correctly identifying a single minority-class example — the metric can't distinguish a genuinely good model from a trivially lazy one. Precision, recall, and F1 (or MCC) are all sensitive to how the minority class specifically is handled, which is exactly the information accuracy discards." />
      <QA q="When would you prefer PR-AUC over ROC-AUC?"
          a="When the positive class is rare and the negative class is huge and not very interesting on its own — fraud detection, rare disease screening. ROC-AUC's False Positive Rate is measured against the large negative class, so it stays high even when precision is actually poor; PR-AUC is directly sensitive to positive-class rarity, giving a more honest picture of how useful the model's positive predictions actually are." />
      <QA q="What's the practical difference between macro-F1 and micro-F1, and when do they diverge the most?"
          a="Macro-F1 averages each class's F1 equally regardless of how many examples that class has; micro-F1 pools every prediction together first, so it's dominated by whichever classes have the most examples. They diverge most when a rare class performs very differently from the common classes — macro will reflect that rare class's struggle strongly, while micro will barely notice it." />
      <QA q="Why is log loss called a 'proper scoring rule' when accuracy and F1 aren't?"
          a="A proper scoring rule is uniquely minimized when the predicted probabilities exactly match the true underlying probabilities — a model has no incentive to report anything other than its honest confidence. Accuracy and F1 only look at the final hard label after thresholding, so a wildly overconfident but occasionally-correct model can score identically to a well-calibrated one; log loss specifically punishes overconfidence and rewards calibration." />
      <QA q="An MCC comes out very close to 0. What does that tell you?"
          a="An MCC near 0 means the classifier is doing no better than random guessing once class imbalance is accounted for — unlike accuracy, which can look deceptively high on an imbalanced dataset even for a useless classifier, MCC uses all four confusion-matrix cells and stays near 0 in exactly that situation, making it much harder to accidentally fool." />
    </div>
  );
}

function SectionRegression() {
  return (
    <div>
      <P>
        Regression metrics all measure "how far off were the predictions" — but they disagree sharply
        on how to punish a large miss versus several small ones, which makes the choice of metric a
        real design decision, not a formality.
      </P>

      <H2 c="MAE, RMSE & MAPE" />
      <Mx block>{`  MAE  = (1/n) Σ |yᵢ − ŷᵢ|             average absolute error — same units as y,
                                        every error weighted equally
  RMSE = √[ (1/n) Σ (yᵢ − ŷᵢ)² ]        squares errors before averaging — a single
                                        large miss contributes disproportionately
  MAPE = (100/n) Σ |yᵢ − ŷᵢ| / |yᵢ|     percentage error — scale-free, easy to
                                        communicate ("off by 8% on average")`}</Mx>
      <Note color="warning" icon="ti-alert-triangle">
        MAPE breaks down whenever <Mx>y</Mx> can be at or near zero — a true value of 1 with a
        prediction of 3 contributes 200% error, while the reverse (true value 3, prediction 1)
        contributes only 67%; the same absolute miss is scored wildly differently depending on which
        direction it goes, and a true value of exactly 0 makes it undefined.
      </Note>

      <H2 c="Try It — Watch RMSE Race Ahead of MAE" />
      <P>9 fixed, ordinary residuals plus one adjustable outlier — everything else about the "model" stays identical.</P>
      <OutlierSensitivityDemo />

      <H2 c="R² and Adjusted R²" />
      <Mx block>{`  R² = 1 − (SS_res / SS_tot) = 1 − [ Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)² ]

  R² = 1   → the model explains all the variance (perfect fit)
  R² = 0   → the model is no better than always predicting the mean ȳ
  R² < 0   → the model is worse than just predicting the mean (possible! e.g. an
             overfit model evaluated on new data)

  Adjusted R² = 1 − (1−R²)·(n−1)/(n−p−1)      p = number of features`}</Mx>
      <Note color="danger" icon="ti-alert-triangle">
        Plain R² can <em>never decrease</em> when you add any feature to the model, even pure random
        noise — the least-squares fit will always find some infinitesimally small use for an extra
        column, so R² alone can't tell you whether a feature actually helped. Adjusted R² adds an
        explicit penalty for extra features, and can decrease if a new feature doesn't pull its
        weight — a much fairer way to compare models with different numbers of features.
      </Note>

      <H2 c="Huber Loss — and Loss vs. Metric" />
      <P>
        Huber loss behaves like MSE for small residuals and like MAE for large ones (past a threshold
        <Mx>δ</Mx>), giving a robust middle ground: sensitive enough to differentiate small errors
        smoothly (unlike MAE, whose gradient is a constant ±1 everywhere), but not as dominated by
        outliers as MSE. It's almost always described as a <strong>loss function</strong> (what a
        model is trained against) rather than an <strong>evaluation metric</strong> (what's reported
        afterward) — a real, useful distinction: nothing stops training with one function and
        reporting a different one (e.g. train with Huber loss for robustness, but still report RMSE
        and MAE afterward, since those are what stakeholders actually understand).
      </P>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

n, p = len(y_test), X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="You're evaluating a house-price model and one prediction is off by $2 million due to a data-entry error in that one row. Would you rather report MAE or RMSE?"
          a="MAE, if you want a metric that isn't dominated by that single anomalous row — MAE grows linearly with any one error's size, while RMSE's squaring means one huge outlier can dwarf the contribution of every other, otherwise-reasonable prediction. RMSE is the right choice when large errors genuinely deserve outsized weight; MAE is right when you want a robust, typical-case picture." />
      <QA q="What's wrong with using R² alone to decide whether a newly added feature is worth keeping?"
          a="R² mathematically cannot decrease when any feature is added, including pure noise, because least squares will always find some infinitesimal use for an extra dimension of freedom. Comparing R² before and after adding a feature will always look like an improvement or a tie — Adjusted R², which explicitly penalizes additional features, is the fairer comparison." />
      <QA q="Why is MAPE risky to use as a primary metric for a target variable that can be close to zero?"
          a="MAPE divides each error by the true value, so as the true value approaches zero the percentage error blows up toward infinity even for a small absolute miss, and it's undefined outright at exactly zero. It also penalizes over-predictions and under-predictions asymmetrically for the same absolute error, which can make model comparisons misleading." />
      <QA q="Can you just use MSE as both the training loss and the reported evaluation metric — why would you ever use two different functions?"
          a="You can, and often people do, but nothing requires it — a model can be trained against one function (e.g. Huber loss, for robustness to outliers during optimization) while being evaluated and reported with a different, more interpretable one (e.g. RMSE and MAE, which stakeholders understand in the target's actual units). The loss just needs to be a good optimization target; the metric needs to be a good communication tool — those aren't always the same function." />
    </div>
  );
}

function SectionUnsupervised() {
  return (
    <div>
      <P>
        Clustering and dimensionality reduction have no ground-truth labels to check predictions
        against — so their metrics instead measure internal properties of the result itself: how
        tightly grouped clusters are, or how much of the original signal survived compression.
      </P>

      <H2 c="Silhouette Score" />
      <Mx block>{`  For point i:  a(i) = mean distance to other points in i's own cluster
                b(i) = mean distance to the nearest OTHER cluster's points

  silhouette(i) = (b(i) − a(i)) / max(a(i), b(i))        range: −1 to +1

  near +1  →  well inside its own cluster, far from the next-nearest one
  near  0  →  sitting right on the boundary between two clusters
  near −1  →  probably assigned to the wrong cluster entirely`}</Mx>
      <P>
        A worked example makes the boundary case concrete: two clean 3-point clusters, plus one
        deliberately borderline point placed almost exactly between them.
      </P>
      <Table
        heads={['Point', 'Cluster', 'Silhouette score']}
        rows={[
          ['Clean cluster-A points', 'A', '0.706 – 0.789'],
          ['Clean cluster-B points', 'B', '0.736 – 0.738'],
          ['Borderline point (3, 2.5)', 'A (by nearest assignment)', '0.090 — right on the fence'],
        ]}
      />

      <H2 c="Davies-Bouldin Index" />
      <Mx block>{`  DB = (1/k) Σᵢ maxⱼ≠ᵢ [ (Sᵢ + Sⱼ) / Mᵢⱼ ]

  Sᵢ  = average distance of cluster i's points to their own centroid (spread)
  Mᵢⱼ = distance between cluster i and j's centroids (separation)

  Lower is better — it's an average of "worst-case" spread-over-separation
  ratios, so one badly-separated pair of clusters dominates the score.`}</Mx>
      <P>
        On the same worked example: the clean two-cluster case scores{' '}
        <strong>DB = 0.308</strong>; folding the borderline point into cluster A (increasing A's
        internal spread without changing the separation between centroids much) raises it to{' '}
        <strong>DB = 0.443</strong> — concretely showing "higher DBI = worse clustering" with the same
        toy dataset used above for silhouette, reinforcing that the two metrics are measuring the same
        underlying idea (tight, well-separated clusters) from different angles.
      </P>

      <H2 c="Elbow Method & Explained Variance Ratio" />
      <P>
        Both already appeared in earlier modules in their natural home, and are named here mainly to
        complete the picture: the <strong>elbow method</strong> (plotting within-cluster sum of
        squares against <Mx>k</Mx> and looking for where the curve's improvement flattens out) is
        covered where it's actually used, in Clustering; <strong>explained variance ratio</strong>
        (how much of the original variance each retained principal component preserves) is covered in
        Dimensionality Reduction.
      </P>

      <H2 c="Two More Worth Knowing By Name" />
      <Table
        heads={['Metric', 'What it measures']}
        rows={[
          ['Calinski-Harabasz Index', 'Ratio of between-cluster to within-cluster dispersion — higher is better; computationally cheaper than silhouette since it only needs cluster centroids, not all pairwise distances.'],
          ['Adjusted Rand Index (ARI)', 'For the rarer case where you DO have ground-truth cluster labels available for comparison — measures agreement between the clustering and the true labels, corrected for chance (like Cohen\'s Kappa, but for cluster assignments instead of classifications).'],
        ]}
      />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.metrics import silhouette_score, davies_bouldin_score

labels = kmeans.fit_predict(X)
print("Silhouette:", silhouette_score(X, labels))
print("Davies-Bouldin:", davies_bouldin_score(X, labels))   # lower is better`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="How can you evaluate whether a clustering result is any good without any ground-truth labels?"
          a="Internal validation metrics like silhouette score and the Davies-Bouldin index score the clustering purely from the data and the cluster assignments themselves — how tight each cluster is internally, and how well-separated it is from its neighbors — without needing to know what the 'true' groups were, since for genuinely unsupervised clustering there usually isn't a true answer to check against." />
      <QA q="What does a silhouette score near zero for a specific point actually mean?"
          a="That point's average distance to its own cluster is roughly equal to its average distance to the next-nearest cluster — it's sitting right on the boundary between two clusters, and a slightly different clustering or distance metric could plausibly have assigned it to either one." />
      <QA q="If you DO happen to have ground-truth labels available, would you still reach for silhouette score?"
          a="No — with true labels available, use a metric designed to compare a clustering against them directly, like the Adjusted Rand Index, which measures agreement corrected for chance. Silhouette score is specifically for the much more common situation where no ground truth exists at all." />
      <QA q="Why does explained variance ratio matter when deciding how many PCA components to keep?"
          a="It directly quantifies the trade-off being made: keeping components that together explain 95% of the variance means only 5% of the original signal is being discarded for the sake of compression, giving a principled, interpretable stopping point instead of picking a number of components arbitrarily." />
    </div>
  );
}

const SECTION_MAP = {
  classification: <SectionClassification />,
  regression: <SectionRegression />,
  unsupervised: <SectionUnsupervised />,
};

export default function Metrics() {
  const [active, setActive] = useState('classification');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 16</div>
        <h1 className="page-header-title">Metrics</h1>
        <p className="page-header-subtitle">
          Classification and regression metrics for supervised models, plus the clustering and
          dimensionality-reduction metrics unsupervised learning needs since it has no labels to
          score against.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={16} />
    </div>
  );
}
