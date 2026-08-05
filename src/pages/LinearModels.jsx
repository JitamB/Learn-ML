import { useState, useRef } from 'react';
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
function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 12px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}

/* ── Linear Regression: draggable trendline demo ───────────── */
const LR_POINTS = (() => {
  const rand = seededRandom(21);
  const n = 18;
  return Array.from({ length: n }, (_, i) => {
    const x = (i + 0.5) / n * 10;
    const noise = (rand() - 0.5) * 4;
    return { x, y: 0.8 * x + 2 + noise };
  });
})();
const LR_N = LR_POINTS.length;
const LR_MEAN_X = LR_POINTS.reduce((s, p) => s + p.x, 0) / LR_N;
const LR_MEAN_Y = LR_POINTS.reduce((s, p) => s + p.y, 0) / LR_N;
const LR_OLS_M = LR_POINTS.reduce((s, p) => s + (p.x - LR_MEAN_X) * (p.y - LR_MEAN_Y), 0)
  / LR_POINTS.reduce((s, p) => s + (p.x - LR_MEAN_X) ** 2, 0);
const LR_OLS_B = LR_MEAN_Y - LR_OLS_M * LR_MEAN_X;

const LR_X_MIN = 0, LR_X_MAX = 10, LR_Y_MIN = -3, LR_Y_MAX = 13;
const LR_SVG_W = 420, LR_SVG_H = 220;
const LR_PAD_L = 34, LR_PAD_R = 14, LR_PAD_T = 14, LR_PAD_B = 26;
const LR_PLOT_W = LR_SVG_W - LR_PAD_L - LR_PAD_R;
const LR_PLOT_H = LR_SVG_H - LR_PAD_T - LR_PAD_B;
const lrXToPx = x => LR_PAD_L + (x - LR_X_MIN) / (LR_X_MAX - LR_X_MIN) * LR_PLOT_W;
const lrYToPx = y => LR_PAD_T + (1 - (y - LR_Y_MIN) / (LR_Y_MAX - LR_Y_MIN)) * LR_PLOT_H;
const lrPxToY = py => LR_Y_MIN + (1 - (py - LR_PAD_T) / LR_PLOT_H) * (LR_Y_MAX - LR_Y_MIN);
const lrClampY = y => Math.min(LR_Y_MAX, Math.max(LR_Y_MIN, y));

function LinearRegressionDemo() {
  const svgRef = useRef(null);
  const [yLeft, setYLeft] = useState(LR_MEAN_Y);
  const [yRight, setYRight] = useState(LR_MEAN_Y);
  const [dragging, setDragging] = useState(null);
  const [showOLS, setShowOLS] = useState(false);

  const m = (yRight - yLeft) / (LR_X_MAX - LR_X_MIN);
  const b = yLeft - m * LR_X_MIN;
  const mse = LR_POINTS.reduce((s, p) => s + (p.y - (m * p.x + b)) ** 2, 0) / LR_N;

  function handlePointerMove(e) {
    if (!dragging || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scaleY = LR_SVG_H / rect.height;
    const py = (e.clientY - rect.top) * scaleY;
    const dataY = lrClampY(lrPxToY(py));
    if (dragging === 'left') setYLeft(dataY); else setYRight(dataY);
  }
  function startDrag(which) {
    return e => { e.target.setPointerCapture(e.pointerId); setDragging(which); };
  }
  function endDrag() { setDragging(null); }

  return (
    <VizBox>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
        <span style={{ fontSize: 12.5, color: 'var(--color-text-secondary)' }}>Drag either endpoint up or down — the line pivots around the other one.</span>
        <button onClick={() => setShowOLS(s => !s)} style={toggleBtnStyle(showOLS)}>{showOLS ? 'Hide' : 'Show'} best fit</button>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${LR_SVG_W} ${LR_SVG_H}`}
        style={{ width: '100%', height: 'auto', display: 'block', touchAction: 'none' }}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
      >
        <line x1={lrXToPx(LR_X_MIN)} y1={lrYToPx(LR_Y_MIN)} x2={lrXToPx(LR_X_MAX)} y2={lrYToPx(LR_Y_MIN)} stroke="var(--color-border-tertiary)" strokeWidth={1} />
        <line x1={lrXToPx(LR_X_MIN)} y1={lrYToPx(LR_Y_MIN)} x2={lrXToPx(LR_X_MIN)} y2={lrYToPx(LR_Y_MAX)} stroke="var(--color-border-tertiary)" strokeWidth={1} />

        {showOLS && (
          <line
            x1={lrXToPx(LR_X_MIN)} y1={lrYToPx(LR_OLS_M * LR_X_MIN + LR_OLS_B)}
            x2={lrXToPx(LR_X_MAX)} y2={lrYToPx(LR_OLS_M * LR_X_MAX + LR_OLS_B)}
            stroke="var(--color-border-success)" strokeWidth={2} strokeDasharray="5,4"
          />
        )}

        <line x1={lrXToPx(LR_X_MIN)} y1={lrYToPx(yLeft)} x2={lrXToPx(LR_X_MAX)} y2={lrYToPx(yRight)} stroke="var(--accent)" strokeWidth={2.5} />

        {LR_POINTS.map((p, i) => (
          <circle key={i} cx={lrXToPx(p.x)} cy={lrYToPx(p.y)} r={4} fill="var(--color-border-info)" opacity={0.85} />
        ))}

        <circle cx={lrXToPx(LR_X_MIN)} cy={lrYToPx(yLeft)} r={7} fill="var(--color-background-primary)" stroke="var(--accent)" strokeWidth={2}
          style={{ cursor: 'ns-resize' }} onPointerDown={startDrag('left')} />
        <circle cx={lrXToPx(LR_X_MAX)} cy={lrYToPx(yRight)} r={7} fill="var(--color-background-primary)" stroke="var(--accent)" strokeWidth={2}
          style={{ cursor: 'ns-resize' }} onPointerDown={startDrag('right')} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 12.5, flexWrap: 'wrap', gap: 8 }}>
        <span style={{ color: 'var(--color-text-secondary)', fontFamily: 'var(--font-mono)' }}>m = {m.toFixed(3)}, b = {b.toFixed(3)}</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--color-text-primary)' }}>MSE = {mse.toFixed(3)}</span>
      </div>
      {showOLS && (
        <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 6 }}>
          Best fit (OLS): m = {LR_OLS_M.toFixed(3)}, b = {LR_OLS_B.toFixed(3)} — the minimum possible MSE for this dataset.
        </div>
      )}
    </VizBox>
  );
}

/* ── Logistic Regression: threshold-shaded decision region ─── */
const LOGREG_W1 = 1, LOGREG_W2 = 1, LOGREG_B = -10;
const sigmoid = z => 1 / (1 + Math.exp(-z));
const LOGREG_POINTS = (() => {
  const rand = seededRandom(7);
  return Array.from({ length: 40 }, () => {
    const x = rand() * 10, y = rand() * 10;
    const noise = (rand() - 0.5) * 4;
    return { x, y, label: (x + y + noise > 10) ? 1 : 0 };
  });
})();
const GRID_COLS = 20, GRID_ROWS = 14;

function LogisticRegressionDemo() {
  const [threshold, setThreshold] = useState(0.5);
  const cells = [];
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const px = (c + 0.5) / GRID_COLS * 10;
      const py = 10 - (r + 0.5) / GRID_ROWS * 10;
      cells.push(sigmoid(LOGREG_W1 * px + LOGREG_W2 * py + LOGREG_B) > threshold ? 1 : 0);
    }
  }

  return (
    <VizBox>
      <SliderRow label="Threshold" min={0.01} max={0.99} step={0.01} value={threshold} onChange={setThreshold} fmt={v => v.toFixed(2)} />
      <div style={{ position: 'relative', height: 220, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${GRID_ROWS}, 1fr)` }}>
          {cells.map((cls, i) => (
            <div key={i} style={{ background: cls ? 'var(--color-background-danger)' : 'var(--color-background-info)' }} />
          ))}
        </div>
        {LOGREG_POINTS.map((p, i) => (
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
        Shaded region = where this already-fitted model predicts class 1 at threshold {threshold.toFixed(2)}. ✕ = actual class 1, ○ = actual
        class 0 — watch how many end up on the "wrong" side of the shading as the threshold moves away from 0.5.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'linreg', label: 'Linear Regression', sub: 'Fitting a line by minimizing RSS' },
  { id: 'logreg', label: 'Logistic Regression', sub: 'Sigmoid, probability & threshold' },
];

function SectionLinearRegression() {
  return (
    <div>
      <P>
        Linear Regression fits a straight line (or, with more than one feature, a hyperplane) that
        minimizes the total squared distance between its predictions and the real values. It's the
        base case nearly every other supervised algorithm gets compared against.
      </P>

      <H2 c="The Model & the Loss" />
      <Mx block>{`  Prediction:          ŷ = w·x + b            (1 feature: ŷ = mx + b)

  Residual Sum of Squares:   RSS = Σᵢ (yᵢ - ŷᵢ)²
  Mean Squared Error:        MSE = RSS / n

  Closed-form solution (Ordinary Least Squares) minimizes RSS exactly —
  no iterative optimization needed for plain linear regression.`}</Mx>

      <H2 c="Core Assumptions" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Linearity">The true relationship between features and target is (approximately) a straight line/hyperplane.</Card>
        <Card color="success" title="Independence of errors">One observation's residual doesn't predict another's — no autocorrelation, common in time series.</Card>
        <Card color="warning" title="Homoscedasticity">Residual variance stays constant across the range of predictions, rather than fanning out.</Card>
        <Card color="purple" title="Normality of residuals">Residuals are approximately normally distributed — mainly matters for the validity of confidence intervals, not the fit itself.</Card>
      </Grid>
      <Note color="info" icon="ti-info-circle">
        These assumptions matter most for trusting a model's <em>inference</em> (p-values, confidence
        intervals). A linear model can still be a useful <em>predictor</em> even when one is mildly
        violated — check residual plots rather than rejecting the model on principle.
      </Note>

      <H2 c="Try It — Fit the Line by Eye" />
      <P>Drag the two endpoints to change the line's slope and intercept — watch the MSE update live, then compare against the true best fit.</P>
      <LinearRegressionDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)          # solves for w, b via OLS

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficients:", model.coef_, "Intercept:", model.intercept_)`}</Code>

      <Note color="success" icon="ti-arrow-right">
        Plain OLS has no penalty on the size of its coefficients — Ridge, Lasso, and Elastic Net add
        one. They're covered in "Regularization" under Model Evaluation & Validation.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why does Ordinary Least Squares have a closed-form solution while most ML training needs iterative optimization?"
          a="RSS is a convex quadratic function of the weights, so its gradient is linear in the weights — setting that gradient to zero and solving is just linear algebra (the normal equation), with a single global minimum. Most other loss functions (log-loss, hinge loss, anything wrapped around a neural network) aren't quadratic, so there's no algebraic shortcut and gradient-based iteration is the only practical option." />
      <QA q="What does it mean for a linear regression's residuals to be heteroscedastic, and why does it matter?"
          a="Heteroscedasticity means the spread of residuals changes across the range of predicted values — e.g. errors grow larger for bigger predicted house prices. The coefficient estimates themselves stay unbiased, but the standard errors (and therefore p-values and confidence intervals) become unreliable, which matters if you're using the model for inference rather than just prediction." />
      <QA q="A feature has a strong non-linear (e.g. quadratic) relationship with the target. Does that rule out linear regression?"
          a="No — linear regression only needs to be linear in its parameters, not in the raw features. Adding a transformed feature (e.g. x²) as an extra column lets the same OLS machinery fit a curved relationship; this is exactly what polynomial regression is." />
    </div>
  );
}

function SectionLogisticRegression() {
  return (
    <div>
      <P>
        Logistic Regression wraps the same linear equation in a sigmoid function, squashing its
        output into a probability between 0 and 1 — used for binary classification (spam vs. not
        spam, churn vs. not churn).
      </P>

      <H2 c="The Sigmoid & Decision Rule" />
      <Mx block>{`  Linear score:   z = w·x + b
  Sigmoid:        σ(z) = 1 / (1 + e^(-z))          — squashes z into (0, 1)

  Predict class 1 if σ(z) > threshold      (threshold defaults to 0.5)

  Trained by minimizing log loss (cross-entropy), not RSS — RSS on top of
  a sigmoid isn't convex, log loss is.`}</Mx>

      <H2 c="Moving the Threshold" />
      <P>
        The threshold isn't sacred — it's a knob. Raising it makes the model more conservative about
        predicting class 1 (fewer false positives, more false negatives); lowering it does the
        opposite. Where to set it depends on which mistake is more costly for the problem at hand.
      </P>

      <H2 c="Try It — Sweep the Threshold" />
      <P>Same 40 labeled points and the same fitted model — only the threshold changes.</P>
      <LogisticRegressionDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

clf = LogisticRegression()
clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]     # P(class = 1)
custom_preds = (probs > 0.3).astype(int)    # move the threshold manually
print(confusion_matrix(y_test, custom_preds))`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why is log loss used to train logistic regression instead of mean squared error?"
          a="Plugging a sigmoid into MSE produces a non-convex loss surface with multiple local minima, so gradient descent isn't guaranteed to find the best fit. Log loss (cross-entropy) stays convex when paired with a sigmoid, guaranteeing a single global minimum that standard gradient-based solvers can reliably reach." />
      <QA q="You lower a fraud classifier's decision threshold from 0.5 to 0.2. What happens to precision and recall?"
          a="Recall goes up and precision goes down. A lower threshold means the model flags more cases as fraud (catching more true fraud — higher recall) but also flags more legitimate transactions by mistake (more false positives — lower precision). The right threshold depends on the relative cost of missing fraud versus annoying a legitimate customer." />
      <QA q="Is logistic regression a linear or non-linear model?"
          a="Its decision boundary is linear — the sigmoid is a monotonic squashing function applied after the linear combination, so σ(w·x+b) > threshold reduces back to a linear inequality in x. The output probability is a non-linear function of the inputs, but the boundary between predicted classes is still a straight line (or hyperplane)." />
    </div>
  );
}

const SECTION_MAP = {
  linreg: <SectionLinearRegression />,
  logreg: <SectionLogisticRegression />,
};

export default function LinearModels() {
  const [active, setActive] = useState('linreg');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 04</div>
        <h1 className="page-header-title">Linear Models</h1>
        <p className="page-header-subtitle">
          Fit a line to predict a number, or a sigmoid-wrapped line to predict a class probability —
          the base case nearly every other supervised algorithm builds on.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={4} />
    </div>
  );
}
