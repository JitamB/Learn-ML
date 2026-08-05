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
function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 12px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}
const GRID_COLS = 20, GRID_ROWS = 14;

/* ── SVM demo: real Kernel-Adatron solver ──────────────────── */
const SVM_POINTS = (() => {
  const rand = seededRandom(7);
  const pts = [];
  for (let i = 0; i < 25; i++) {
    const a = rand() * 2 * Math.PI, r = rand() * 2.0;
    pts.push({ x: 5 + r * Math.cos(a), y: 5 + r * Math.sin(a), label: -1 });
  }
  for (let i = 0; i < 25; i++) {
    const a = rand() * 2 * Math.PI, r = 3.2 + rand() * 1.3;
    pts.push({ x: 5 + r * Math.cos(a), y: 5 + r * Math.sin(a), label: 1 });
  }
  return pts;
})();
function kernelLinear(a, b) { return a.x * b.x + a.y * b.y + 1; }
function kernelRBF(a, b, gamma) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.exp(-gamma * (dx * dx + dy * dy)) + 1;
}
const SVM_ITERS = 150, SVM_ETA = 0.1;

function SVMDemo() {
  const [kernelType, setKernelType] = useState('rbf');
  const [gammaRaw, setGammaRaw] = useState(-0.7); // 10^-0.7 ≈ 0.20
  const [cRaw, setCRaw] = useState(0);            // 10^0 = 1.00
  const gamma = 10 ** gammaRaw;
  const C = 10 ** cRaw;
  const kernelFn = (a, b) => (kernelType === 'rbf' ? kernelRBF(a, b, gamma) : kernelLinear(a, b));

  // Kernel matrix only depends on kernel choice + gamma — cached separately from C so dragging
  // C alone doesn't pay to rebuild it.
  const K = useMemo(
    () => SVM_POINTS.map(pi => SVM_POINTS.map(pj => kernelFn(pi, pj))),
    [kernelType, gamma]
  );
  // Kernel-Adatron: an iterative approximate solver for the SVM dual (Frieß, Cristianini &
  // Campbell, 1998) — simplified vs. sklearn's actual SMO solver, but a real algorithm: the
  // box constraint [0, C] below is where C genuinely changes the fitted decision function.
  const alpha = useMemo(() => {
    const n = SVM_POINTS.length;
    const a = new Array(n).fill(0);
    for (let t = 0; t < SVM_ITERS; t++) {
      for (let i = 0; i < n; i++) {
        let z = 0;
        for (let j = 0; j < n; j++) z += a[j] * SVM_POINTS[j].label * K[i][j];
        a[i] = Math.max(0, Math.min(C, a[i] + SVM_ETA * (1 - SVM_POINTS[i].label * z)));
      }
    }
    return a;
  }, [K, C]);

  function score(x, y) {
    let z = 0;
    for (let j = 0; j < SVM_POINTS.length; j++) z += alpha[j] * SVM_POINTS[j].label * kernelFn(SVM_POINTS[j], { x, y });
    return z;
  }

  const cells = [];
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const px = (c + 0.5) / GRID_COLS * 10;
      const py = 10 - (r + 0.5) / GRID_ROWS * 10;
      cells.push(score(px, py) > 0 ? 1 : 0);
    }
  }
  const trainAcc = SVM_POINTS.filter(p => (score(p.x, p.y) > 0 ? 1 : -1) === p.label).length / SVM_POINTS.length * 100;
  const supportVectors = alpha.filter(a => a > 1e-3).length;
  const boundSVs = alpha.filter(a => a > C - 1e-3).length;

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        {[{ id: 'linear', label: 'Linear' }, { id: 'rbf', label: 'RBF' }].map(k => (
          <button key={k.id} onClick={() => setKernelType(k.id)} style={toggleBtnStyle(kernelType === k.id)}>{k.label}</button>
        ))}
      </div>
      <SliderRow label="Gamma" min={-1.3} max={0} step={0.02} value={gammaRaw} onChange={setGammaRaw} fmt={() => gamma.toFixed(2)} />
      <SliderRow label="C" min={-1.3} max={0.5} step={0.02} value={cRaw} onChange={setCRaw} fmt={() => C.toFixed(2)} />
      <div style={{ position: 'relative', height: 220, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', inset: 0, display: 'grid', gridTemplateColumns: `repeat(${GRID_COLS}, 1fr)`, gridTemplateRows: `repeat(${GRID_ROWS}, 1fr)` }}>
          {cells.map((cls, i) => (
            <div key={i} style={{ background: cls ? 'var(--color-background-danger)' : 'var(--color-background-info)' }} />
          ))}
        </div>
        {SVM_POINTS.map((p, i) => (
          <div key={i} style={{
            position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
            transform: 'translate(-50%,-50%)', fontSize: 12, fontWeight: 700, lineHeight: 1,
            color: p.label === 1 ? 'var(--color-text-danger)' : 'var(--color-text-info)',
          }}>
            {p.label === 1 ? '✕' : '○'}
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        Train accuracy: {trainAcc.toFixed(0)}% · {supportVectors} support vectors ({boundSVs} pinned at the C ceiling).{' '}
        {kernelType === 'linear'
          ? "A ring can't be separated by a straight line — no amount of tuning C fixes that."
          : 'RBF can wrap around the inner ring — Gamma controls how tightly it hugs individual points.'}
      </div>
    </VizBox>
  );
}

/* ── Naive Bayes demo: prior slider shifting the posterior ──── */
function gaussPdf(x, mean, std) {
  return Math.exp(-0.5 * ((x - mean) / std) ** 2) / (std * Math.sqrt(2 * Math.PI));
}
const NB_MEAN_A = 30, NB_STD_A = 8, NB_MEAN_B = 55, NB_STD_B = 10;
const NB_BARS = 50;

function NaiveBayesDemo() {
  const [priorA, setPriorA] = useState(0.5);
  const bars = Array.from({ length: NB_BARS }, (_, i) => {
    const x = (i + 0.5) / NB_BARS * 100;
    return { pdfA: gaussPdf(x, NB_MEAN_A, NB_STD_A), pdfB: gaussPdf(x, NB_MEAN_B, NB_STD_B) };
  });
  const maxPdf = Math.max(...bars.map(b => Math.max(b.pdfA, b.pdfB)));

  // Scan the full visible 0-100 axis for the posterior crossing. Note: unequal variances make the
  // log-posterior-odds quadratic in x, so a second mathematical root exists — verified far off-screen
  // (x ≈ -53 to -84) across the whole prior range for these exact means/stds. Re-verify if they change.
  let crossing = null, prevSign = null;
  for (let i = 0; i <= 200; i++) {
    const x = (i / 200) * 100;
    const f = priorA * gaussPdf(x, NB_MEAN_A, NB_STD_A) - (1 - priorA) * gaussPdf(x, NB_MEAN_B, NB_STD_B);
    const sign = f >= 0;
    if (prevSign !== null && sign !== prevSign) { crossing = x; break; }
    prevSign = sign;
  }

  return (
    <VizBox>
      <SliderRow label="P(class A)" min={0.01} max={0.99} step={0.01} value={priorA} onChange={setPriorA} fmt={v => v.toFixed(2)} />
      <div style={{ position: 'relative', height: 140, marginTop: 14 }}>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', gap: 1, borderBottom: '1px solid var(--color-border-tertiary)' }}>
          {bars.map((bar, i) => (
            <div key={i} style={{ flex: 1, position: 'relative', height: '100%' }}>
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${(bar.pdfA / maxPdf) * 100}%`, background: 'var(--color-background-info)', opacity: 0.85 }} />
              <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: `${(bar.pdfB / maxPdf) * 100}%`, background: 'var(--color-background-danger)', opacity: 0.6 }} />
            </div>
          ))}
        </div>
        {crossing !== null && (
          <div style={{ position: 'absolute', left: `${crossing}%`, top: 0, bottom: 0, width: 2, background: 'var(--color-text-primary)' }} />
        )}
      </div>
      <div style={{ display: 'flex', gap: 16, fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8, flexWrap: 'wrap' }}>
        <span><span style={{ color: 'var(--color-text-info)' }}>■</span> Class A likelihood</span>
        <span><span style={{ color: 'var(--color-text-danger)' }}>■</span> Class B likelihood</span>
        <span style={{ marginLeft: 'auto' }}>
          {crossing !== null ? `Posterior boundary at x ≈ ${crossing.toFixed(0)} — left of it classifies as A, right as B.` : 'One class dominates the whole visible axis.'}
        </span>
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'svm', label: 'Support Vector Machines', sub: 'Maximum-margin boundaries' },
  { id: 'nb', label: 'Naive Bayes', sub: "Bayes' theorem, applied naively" },
];

function SectionSVM() {
  return (
    <div>
      <P>
        Support Vector Machines find the boundary between classes that maximizes the margin — the
        distance to the nearest point of either class. Only those nearest points, the{' '}
        <strong>support vectors</strong>, actually determine where the boundary sits.
      </P>

      <H2 c="The Kernel Trick" />
      <P>
        Many real datasets aren't linearly separable in their original feature space. Kernels
        implicitly map points into a higher-dimensional space where a linear boundary <em>can</em>{' '}
        separate them — without ever computing the mapping explicitly.
      </P>
      <Table
        heads={['Kernel', 'Formula', 'When to use']}
        rows={[
          ['Linear', 'K(a,b) = a·b', 'Data is already roughly linearly separable'],
          ['Polynomial', 'K(a,b) = (a·b + c)ᵈ', 'Interactions up to a fixed degree matter'],
          ['RBF (Gaussian)', 'K(a,b) = exp(-γ‖a-b‖²)', 'Unknown, likely non-linear structure — the default first choice'],
        ]}
      />

      <H2 c="Hyperparameters: C and Gamma" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="C — regularization">
          Trades margin width against misclassification tolerance. Low C allows a wider margin with
          some points inside it; high C forces the boundary to hug the training points tightly.
        </Card>
        <Card color="success" title="Gamma — kernel reach (RBF only)">
          How far a single training point's influence reaches. Low gamma → smooth, far-reaching
          boundaries; high gamma → tight boundaries that wrap closely around individual points.
        </Card>
      </Grid>

      <H2 c="Try It — Tune the Boundary" />
      <P>50 points arranged in two rings, one inside the other — a textbook non-linearly-separable case. Switch kernels and drag C/Gamma.</P>
      <SVMDemo />
      <Note color="info" icon="ti-info-circle">
        This demo runs a real, small iterative solver (Kernel-Adatron) so C and Gamma genuinely
        change the fitted boundary — not a decorative overlay. It's a simplified stand-in for the
        Sequential Minimal Optimization (SMO) solver scikit-learn actually uses, which the numbers
        below won't match exactly, but the qualitative behavior (kernel choice, C, and Gamma each
        doing what the theory says) is the same.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.svm import SVC

svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)

print("Number of support vectors:", svm.n_support_)`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does an SVM's decision boundary only depend on the support vectors, not every training point?"
          a="The margin-maximizing objective only cares about the closest points to the boundary — points sitting safely on the correct side, far from the margin, get zero weight (α=0) in the solution regardless of where exactly they sit. This is also why SVM predictions can be fast at inference: only the (often small) set of support vectors need to be stored and evaluated." />
      <QA q="What happens to an SVM's boundary as Gamma (RBF kernel) gets very large?"
          a="Very high gamma makes each training point's influence extremely localized — the boundary starts wrapping tightly around individual points rather than capturing the general shape of the classes, which is overfitting. Very low gamma has the opposite problem: the influence is so broad the boundary becomes almost linear, underfitting genuinely non-linear structure." />
      <QA q="Why can't a linear kernel separate a dataset shaped like one class ringed around another?"
          a="A linear kernel's decision boundary is a straight line (or hyperplane) — there is no single straight line that puts every inner-ring point on one side and every outer-ring point on the other, since the outer ring surrounds the inner one in every direction. A kernel that can express a curved (e.g. circular) boundary, like RBF, is required." />
    </div>
  );
}

function SectionNaiveBayes() {
  return (
    <div>
      <P>
        Naive Bayes classifiers apply Bayes' theorem directly, under one deliberately simplifying
        ("naive") assumption: every feature is independent of every other, given the class.
      </P>

      <H2 c="Bayes' Theorem" />
      <Mx block>{`  P(class | x) = P(x | class) · P(class) / P(x)

  posterior = likelihood × prior / evidence

  "Naive" independence assumption lets P(x | class) factor into a simple
  product across features instead of needing their full joint distribution:

  P(x₁,...,xₙ | class) ≈ P(x₁|class) · P(x₂|class) · ... · P(xₙ|class)`}</Mx>
      <Note color="info" icon="ti-info-circle">
        The independence assumption is almost never literally true, yet Naive Bayes remains a strong,
        fast baseline — for classification, only the <em>ranking</em> of posteriors across classes
        needs to be right, not their exact probability values.
      </Note>

      <H2 c="Choosing a Variant" />
      <Table
        heads={['Variant', 'Best Used For', 'Example Use Case']}
        rows={[
          ['Gaussian NB', 'Continuous data assumed to follow a normal distribution', 'Sensor readings, physical measurements'],
          ['Multinomial NB', 'Discrete counts or frequencies', 'Text classification (bag-of-words / TF-IDF)'],
          ['Bernoulli NB', 'Binary/boolean features', 'Document classification (word presence/absence)'],
        ]}
      />

      <H2 c="Try It — Move the Prior" />
      <P>Two overlapping class likelihoods along one feature — watch the posterior decision point shift as the prior changes.</P>
      <NaiveBayesDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)

probs = nb.predict_proba(X_test)   # posterior P(class | x) per row`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does Naive Bayes often work well in practice despite its independence assumption being unrealistic?"
          a="Classification only requires the posterior probabilities to be ranked correctly across classes, not to be numerically accurate — even when the independence assumption distorts the actual probability values, it frequently distorts them in a way that preserves which class comes out on top, especially for problems like text classification with many weakly-correlated features." />
      <QA q="How does changing the prior P(class) shift the decision boundary, holding the likelihoods fixed?"
          a="The posterior compares prior × likelihood across classes. Raising a class's prior shifts the crossing point of the two posterior curves toward the other class's territory — effectively requiring stronger evidence (a more extreme feature value) before the model will predict the now-less-favored class, even though nothing about the underlying feature distributions changed." />
      <QA q="Why is Multinomial Naive Bayes typically used for text classification instead of Gaussian?"
          a="Text features (word counts or TF-IDF weights) are discrete frequency-like counts, not continuous measurements from a bell-shaped distribution — Multinomial NB's likelihood model matches that structure directly, while Gaussian NB would be assuming a normal distribution that word counts don't actually follow." />
    </div>
  );
}

const SECTION_MAP = {
  svm: <SectionSVM />,
  nb: <SectionNaiveBayes />,
};

export default function SVMNaiveBayes() {
  const [active, setActive] = useState('svm');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 06</div>
        <h1 className="page-header-title">SVM & Naive Bayes</h1>
        <p className="page-header-subtitle">
          Support Vector Machines find the maximum-margin boundary between classes; Naive Bayes
          classifies by applying Bayes' theorem under a (naive) feature-independence assumption.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={6} />
    </div>
  );
}
