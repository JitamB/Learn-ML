import { useState, useMemo } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, VizBox } from '../components/ui/Primitives.jsx';
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
function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 12px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}

/* ── Gradient Boosting demo: real stump-on-residuals fit ────── */
const BOOST_POINTS = (() => {
  const rand = seededRandom(5);
  const f = x => 3 * Math.sin(x * 0.9) + 0.4 * x;
  return Array.from({ length: 20 }, (_, i) => {
    const x = (i + 0.5) / 20 * 10;
    return { x, y: f(x) + (rand() - 0.5) * 1.0 };
  });
})();
const BOOST_LR = 0.5, BOOST_MAX_STAGE = 8;
const BOOST_Y_MIN = Math.min(...BOOST_POINTS.map(p => p.y)) - 1.5;
const BOOST_Y_MAX = Math.max(...BOOST_POINTS.map(p => p.y)) + 1.5;
const BOOST_SVG_W = 460, BOOST_SVG_H = 220;
const BOOST_PAD_L = 32, BOOST_PAD_R = 14, BOOST_PAD_T = 14, BOOST_PAD_B = 24;
const BOOST_PLOT_W = BOOST_SVG_W - BOOST_PAD_L - BOOST_PAD_R;
const BOOST_PLOT_H = BOOST_SVG_H - BOOST_PAD_T - BOOST_PAD_B;
const boostXToPx = x => BOOST_PAD_L + (x / 10) * BOOST_PLOT_W;
const boostYToPx = y => BOOST_PAD_T + (1 - (y - BOOST_Y_MIN) / (BOOST_Y_MAX - BOOST_Y_MIN)) * BOOST_PLOT_H;

function fitStump(xs, residuals) {
  const sortedX = [...new Set(xs)].sort((a, b) => a - b);
  let best = null;
  for (let i = 0; i < sortedX.length - 1; i++) {
    const thresh = (sortedX[i] + sortedX[i + 1]) / 2;
    let sumL = 0, nL = 0, sumR = 0, nR = 0;
    for (let j = 0; j < xs.length; j++) {
      if (xs[j] <= thresh) { sumL += residuals[j]; nL++; } else { sumR += residuals[j]; nR++; }
    }
    if (!nL || !nR) continue;
    const meanL = sumL / nL, meanR = sumR / nR;
    let sse = 0;
    for (let j = 0; j < xs.length; j++) {
      const p = xs[j] <= thresh ? meanL : meanR;
      sse += (residuals[j] - p) ** 2;
    }
    if (!best || sse < best.sse) best = { thresh, meanL, meanR, sse };
  }
  return best;
}

function BoostingDemo() {
  const [stage, setStage] = useState(0);
  const xs = useMemo(() => BOOST_POINTS.map(p => p.x), []);
  const ys = useMemo(() => BOOST_POINTS.map(p => p.y), []);
  const meanY = useMemo(() => ys.reduce((a, b) => a + b, 0) / ys.length, [ys]);

  const { pred, stumps } = useMemo(() => {
    let pred = new Array(xs.length).fill(meanY);
    const stumps = [];
    for (let s = 0; s < stage; s++) {
      const residuals = ys.map((y, i) => y - pred[i]);
      const stump = fitStump(xs, residuals);
      stumps.push(stump);
      pred = pred.map((p, i) => p + BOOST_LR * (xs[i] <= stump.thresh ? stump.meanL : stump.meanR));
    }
    return { pred, stumps };
  }, [stage, xs, ys, meanY]);

  const sse = ys.reduce((s, y, i) => s + (y - pred[i]) ** 2, 0);
  const initialSSE = ys.reduce((s, y) => s + (y - meanY) ** 2, 0);
  const stepDisabled = stage >= BOOST_MAX_STAGE;

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
        <button
          onClick={() => setStage(s => Math.min(BOOST_MAX_STAGE, s + 1))}
          disabled={stepDisabled}
          style={{ ...toggleBtnStyle(!stepDisabled), opacity: stepDisabled ? 0.4 : 1, cursor: stepDisabled ? 'default' : 'pointer' }}
        >
          Step →
        </button>
        <button onClick={() => setStage(0)} style={toggleBtnStyle(false)}>Reset</button>
        <span style={{ fontSize: 12.5, color: 'var(--color-text-secondary)', marginLeft: 'auto', fontFamily: 'var(--font-mono)' }}>
          Stage {stage} / {BOOST_MAX_STAGE}
        </span>
      </div>
      <svg viewBox={`0 0 ${BOOST_SVG_W} ${BOOST_SVG_H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <line x1={boostXToPx(0)} y1={boostYToPx(BOOST_Y_MIN)} x2={boostXToPx(10)} y2={boostYToPx(BOOST_Y_MIN)} stroke="var(--color-border-tertiary)" strokeWidth={1} />
        <line x1={boostXToPx(0)} y1={boostYToPx(BOOST_Y_MIN)} x2={boostXToPx(0)} y2={boostYToPx(BOOST_Y_MAX)} stroke="var(--color-border-tertiary)" strokeWidth={1} />

        {BOOST_POINTS.map((p, i) => (
          <line key={`res-${i}`} x1={boostXToPx(p.x)} y1={boostYToPx(p.y)} x2={boostXToPx(p.x)} y2={boostYToPx(pred[i])}
            stroke="var(--color-text-danger)" strokeWidth={1.5} opacity={0.5} />
        ))}

        <polyline
          points={BOOST_POINTS.map((p, i) => `${boostXToPx(p.x)},${boostYToPx(pred[i])}`).join(' ')}
          fill="none" stroke="var(--accent)" strokeWidth={2}
        />

        {BOOST_POINTS.map((p, i) => (
          <circle key={`pt-${i}`} cx={boostXToPx(p.x)} cy={boostYToPx(p.y)} r={4} fill="var(--color-border-info)" />
        ))}
      </svg>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        SSE: {sse.toFixed(1)} ({((sse / initialSSE) * 100).toFixed(0)}% of the starting error).{' '}
        {stage === 0
          ? 'Stage 0 — every prediction starts flat, at the training mean.'
          : `Stage ${stage} added a stump splitting near x ≈ ${stumps[stumps.length - 1].thresh.toFixed(2)}, nudging whichever side still had the larger average residual (red lines) toward the real value.`}
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'adaboost', label: 'AdaBoost', sub: 'Adaptive reweighting' },
  { id: 'gbm', label: 'Gradient Boosting', sub: 'Fitting the residuals' },
  { id: 'kaggle', label: 'XGBoost, LightGBM & CatBoost', sub: 'The Kaggle winners' },
  { id: 'cheatsheet', label: 'Choosing an Algorithm', sub: 'A category-wide comparison' },
];

function SectionAdaBoost() {
  return (
    <div>
      <P>
        Boosting builds models sequentially — each new model tries specifically to correct the
        mistakes of the ones built before it — rather than training many independent models like
        bagging does. AdaBoost (Adaptive Boosting) was the original, and still-instructive, version
        of this idea.
      </P>

      <H2 c="How AdaBoost Works" />
      <Mx block>{`  1. Start with equal weights on every training point: wᵢ = 1/n
  2. Fit a weak learner (typically a "stump" — a depth-1 tree) using
     those weights
  3. Compute the weighted error rate ε of that stump
  4. Give the stump a vote-weight:  α = 0.5 · ln((1-ε) / ε)
     (a more accurate stump gets a louder vote)
  5. Increase the weight of points the stump got wrong, decrease it for
     points it got right — so the NEXT stump is forced to focus on
     exactly what this one missed
  6. Repeat for M rounds; final prediction = sign(Σ αₘ · stumpₘ(x))`}</Mx>
      <Note color="info" icon="ti-info-circle">
        The mechanics here — reweighting misclassified points — are specific to classification. The
        next tab demonstrates boosting's general "fit the residuals" idea with a live, real solver on
        a regression problem instead, which is simpler to visualize but carries the same core insight.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),   # a "stump"
    n_estimators=100,
    learning_rate=1.0,
    random_state=42,
)
ada.fit(X_train, y_train)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does AdaBoost specifically use weak learners like depth-1 stumps rather than full trees?"
          a="Boosting's whole mechanism relies on combining many models that are each individually just slightly better than random guessing, then letting sequential correction and weighted voting compound those small edges into a strong learner. A full, low-bias tree would fit each round's reweighted data very closely, leaving the next round little useful signal to correct — and would make the ensemble slow and prone to overfitting the reweighted noise." />
      <QA q="What does AdaBoost's α (alpha) formula tell you about a stump with 50% error?"
          a="α = 0.5·ln((1-ε)/ε) evaluates to 0 when ε=0.5 — a stump doing no better than a coin flip gets zero vote, correctly contributing nothing to the ensemble. As ε drops below 0.5, α grows positive (and unboundedly large as ε→0), meaning increasingly accurate stumps get proportionally louder votes." />
      <QA q="How does boosting's approach to combining models differ fundamentally from bagging's?"
          a="Bagging trains independent models in parallel on resampled data and averages them to cancel out variance. Boosting trains models sequentially, where each new model is explicitly built to fix what previous ones got wrong — it primarily reduces bias, and the models are correlated by design (each depends on the ones before it) rather than independent." />
    </div>
  );
}

function SectionGBM() {
  return (
    <div>
      <P>
        Gradient Boosting generalizes AdaBoost's idea: instead of reweighting misclassified points,
        each new tree is fit directly to the <strong>residual errors</strong> (for regression with
        squared-error loss, these are just the plain leftover errors — more generally, the negative
        gradient of the loss function) of everything built so far.
      </P>

      <H2 c="The Core Idea" />
      <Mx block>{`  F₀(x) = mean(y)                              — start flat, at the mean

  For m = 1 to M:
    residualsᵢ = yᵢ - F_{m-1}(xᵢ)                — what's still wrong
    fit a small tree hₘ(x) to predict the residuals
    F_m(x) = F_{m-1}(x) + learning_rate · hₘ(x)   — shrink the correction

  Final prediction: F_M(x)`}</Mx>
      <Note color="warning" icon="ti-alert-triangle">
        The learning rate matters as much as the number of stages. A high learning rate corrects
        fast but can overshoot and overfit in a few rounds; a low one needs many more rounds but
        generalizes better — the two are tuned together, never in isolation.
      </Note>

      <H2 c="Try It — Step Through the Fit" />
      <P>20 points along a wavy curve. Each click fits one stump to the current residuals (red lines) and shrinks the fit toward them.</P>
      <BoostingDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
gbm.fit(X_train, y_train)`}</Code>

      <Note color="success" icon="ti-arrow-right">
        Learning rate, number of stages, and tree depth all interact and all need systematic
        search — covered in "Hyperparameter Tuning" under Model Evaluation & Validation.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why is the learning rate in gradient boosting sometimes called 'shrinkage'?"
          a="Each new tree's correction is multiplied by the learning rate before being added to the running prediction, deliberately shrinking its contribution below what the tree alone would predict. This forces the ensemble to make many small, conservative corrections instead of a few large ones, which empirically generalizes far better even though it needs more stages to converge." />
      <QA q="If you had to change exactly one hyperparameter to fix a gradient boosting model that's clearly overfitting, which would you reach for first?"
          a="Lowering the learning rate (paired with more estimators to compensate) or reducing max_depth are the two most direct levers — both reduce how aggressively any single tree can fit the current residuals, which is usually the actual source of overfitting in a boosted model, more so than simply reducing n_estimators alone." />
      <QA q="What does it mean that gradient boosting fits new trees to the 'negative gradient of the loss'?"
          a="For squared-error loss, the negative gradient with respect to the current prediction happens to equal the plain residual (y - F(x)), which is why the simple description 'fit the residuals' works for regression. For other loss functions (e.g. log-loss for classification), the negative gradient is a different quantity, but it's always some measure of 'the direction the current prediction needs to move to reduce loss' — which is what ties AdaBoost, gradient boosting, and even boosted classification together as one family." />
    </div>
  );
}

function SectionKaggleWinners() {
  return (
    <div>
      <P>
        Plain gradient boosting is a general recipe — XGBoost, LightGBM, and CatBoost are heavily
        engineered implementations of it, each optimizing a different part of the pipeline. Together
        they dominate tabular-data competitions and a large share of production tabular models.
      </P>

      <H2 c="What Each One Adds" />
      <Table
        heads={['Framework', 'Key Idea', 'Best For']}
        rows={[
          ['XGBoost', 'Adds explicit L1/L2 regularization to the tree-building objective, builds trees with parallelized, cache-aware split-finding', 'A strong, well-tested default — usually the first thing to try'],
          ['LightGBM', 'Histogram-based binning of continuous features plus leaf-wise (rather than level-wise) tree growth', 'Very large datasets where training speed matters most'],
          ['CatBoost', 'Handles categorical features natively via target statistics (no manual encoding needed), uses "ordered boosting" to fight target leakage', 'Data with many categorical columns'],
        ]}
      />

      <H3 c="Basic usage — all three share a scikit-learn-compatible API" />
      <Code>{`import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
xgb_model.fit(X_train, y_train)

lgb_model = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05)
lgb_model.fit(X_train, y_train)

# CatBoost can take raw categorical columns directly — no one-hot/label encoding needed
cat_model = CatBoostClassifier(iterations=300, depth=6, verbose=False)
cat_model.fit(X_train, y_train, cat_features=["city", "device_type"])`}</Code>

      <Note color="info" icon="ti-info-circle">
        "Leaf-wise" (LightGBM) growth picks whichever leaf reduces loss the most next, regardless of
        depth — often more accurate per tree than "level-wise" growth (XGBoost's default), but more
        prone to overfitting on small datasets since it can produce a few very deep, narrow branches.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why is LightGBM typically faster than XGBoost on very large datasets?"
          a="LightGBM bins continuous features into a fixed number of discrete buckets (a histogram) before searching for splits, turning an O(n) scan per candidate threshold into a much cheaper O(bins) scan. Combined with leaf-wise growth (always expanding the single most-improving leaf next, rather than every leaf at the current depth), it typically needs less computation to reach a comparable fit." />
      <QA q="What problem does CatBoost's 'ordered boosting' solve?"
          a="Naively computing target statistics for categorical encoding (e.g. mean target per category) using the full training set leaks target information into the features used to predict that same target — a subtle form of overfitting. Ordered boosting computes each row's category statistics using only rows that came before it in a random ordering, avoiding that leak." />
      <QA q="If a dataset has many high-cardinality categorical columns, which of the three frameworks needs the least manual preprocessing?"
          a="CatBoost — it accepts raw categorical columns directly and handles the encoding internally via target statistics. XGBoost and LightGBM both expect numeric input, so categorical columns need to be one-hot, label, or target-encoded beforehand, same as most other models." />
    </div>
  );
}

const CHEATSHEET_ROWS = [
  ['Linear Regression', 'High', 'Very fast', 'No', 'Yes', 'Yes', 'Predicting a number from roughly-linear features'],
  ['Logistic Regression', 'High', 'Very fast', 'No (linear boundary)', 'Yes', 'Somewhat', 'Binary classification with an interpretable boundary'],
  ['K-Nearest Neighbors', 'Medium', 'Instant train, slow predict', 'Yes', 'Yes', 'Yes', 'Small datasets, simple non-linear structure'],
  ['Decision Tree', 'High', 'Fast', 'Yes', 'No', 'No', 'When interpretability of individual predictions matters'],
  ['SVM (RBF)', 'Low', 'Slow on large n', 'Yes', 'Yes', 'Somewhat', 'Small-to-medium datasets with a clear margin'],
  ['Naive Bayes', 'Medium', 'Very fast', 'No (per-feature)', 'No', 'Somewhat', 'Text classification, high-dimensional sparse features'],
  ['Random Forest', 'Medium', 'Moderate', 'Yes', 'No', 'No', 'A strong, low-tuning-effort default for tabular data'],
  ['Gradient Boosting', 'Low', 'Slower to train, fast predict', 'Yes', 'No', 'No', 'Squeezing out maximum accuracy on tabular data'],
];

function SectionCheatSheet() {
  return (
    <div>
      <P>
        A quick reference across every algorithm covered in this category. "Interpretability" means
        whether a human can easily explain a single prediction; the rest are practical properties
        that usually decide which algorithm is worth trying first for a given dataset.
      </P>

      <H2 c="Algorithm Comparison" />
      <Table
        heads={['Algorithm', 'Interpretability', 'Training Speed', 'Handles Non-linearity', 'Sensitive to Scaling', 'Sensitive to Outliers', 'Typical Use Case']}
        rows={CHEATSHEET_ROWS}
      />

      <Note color="success" icon="ti-bulb">
        A common, reasonable default workflow: start with a fast, interpretable baseline (Linear/
        Logistic Regression or a single Decision Tree) to sanity-check the problem and data, then
        reach for Random Forest or Gradient Boosting once a stronger baseline is needed — save SVM
        and KNN for smaller datasets where their weaknesses (training/prediction speed, scaling
        sensitivity) matter less.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="A dataset has 2 million rows and 50 features, and the team needs a first model within a day. What would you rule out immediately, and why?"
          a="SVM with an RBF kernel and plain KNN both scale poorly with dataset size — SVM training complexity grows worse than linearly with n, and KNN's prediction cost scales with the full stored dataset every time. A linear model or a histogram-based gradient boosting framework (LightGBM) would be far more practical starting points at this scale." />
      <QA q="Why might you deliberately choose a less accurate but more interpretable model in practice?"
          a="Regulatory or business requirements (credit decisions, medical diagnoses) sometimes legally or practically require explaining individual predictions — a single decision tree or a linear model's coefficients are far easier to justify to a regulator or a customer than a 300-tree gradient boosting ensemble's output, even if the ensemble scores a few points higher on a test set." />
    </div>
  );
}

const SECTION_MAP = {
  adaboost: <SectionAdaBoost />,
  gbm: <SectionGBM />,
  kaggle: <SectionKaggleWinners />,
  cheatsheet: <SectionCheatSheet />,
};

export default function Boosting() {
  const [active, setActive] = useState('adaboost');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 08</div>
        <h1 className="page-header-title">Boosting</h1>
        <p className="page-header-subtitle">
          Build models sequentially, each one correcting the last one's mistakes — the workhorse
          family behind most tabular-data competition wins.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={8} />
    </div>
  );
}
