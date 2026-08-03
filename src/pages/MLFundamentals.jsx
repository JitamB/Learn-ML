import { useState } from 'react';
import Code from '../components/ui/Code.jsx';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, Badge, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

const TABS = [
  { id: 'def',      label: 'What is ML?',    sub: 'Definitions & the AI landscape' },
  { id: 'types',    label: 'Types of Learning', sub: 'Supervised, unsupervised, RL' },
  { id: 'workflow', label: 'The ML Workflow', sub: 'Data to deployed model' },
  { id: 'mldl',     label: 'ML vs Deep Learning', sub: 'When to use which' },
];

/* ─────────────────── Section: What is ML? ──────────────────── */
function SectionDef() {
  return (
    <div>
      <P>
        Machine Learning (ML) is the field of study that gives computers the ability to find patterns
        in data and improve at a task <em>without being explicitly programmed</em> for every rule.
        Instead of hand-coding logic, we write an algorithm that fits a model's parameters to examples.
      </P>

      <Grid cols={3} gap={12}>
        <Card color="info" title="Artificial Intelligence">Simulating human-like intelligence broadly — rules, search, learning, reasoning.</Card>
        <Card color="success" title="Machine Learning">Learning patterns and decision rules from data instead of hand-coding them.</Card>
        <Card color="purple" title="Deep Learning">A subset of ML using multi-layer neural networks to learn features automatically.</Card>
      </Grid>

      <H2 c="Formal definitions" />
      <P>
        Arthur Samuel (1959) described ML informally as giving computers <em>"the ability to learn
        without being explicitly programmed."</em> Tom Mitchell's 1997 definition is the one most
        textbooks and interviews still use, because it's precise enough to test against:
      </P>
      <Mx block>{`  Tom Mitchell (1997):
  A computer program is said to learn from experience E with respect to
  some class of tasks T and performance measure P, if its performance at
  tasks in T, as measured by P, improves with experience E.`}</Mx>
      <P>
        Example — a spam filter: T = classify emails as spam/not-spam, E = a labeled set of emails,
        P = classification accuracy on new emails. "Learning" means accuracy on P improves as E grows.
      </P>

      <H2 c="Rule-based systems vs. learned systems" />
      <Table
        heads={['', 'Traditional programming', 'Machine learning']}
        rows={[
          ['Input',  'Rules + Data',            'Data + Answers (labels)'],
          ['Output', 'Answers',                  'Rules (a model)'],
          ['Fails when…', 'The world has too many edge cases to enumerate', 'There isn’t enough representative data'],
        ]}
      />
      <Note color="info" icon="ti-info-circle">
        Flip the inputs and outputs: a traditional program takes rules and data to produce answers.
        An ML system takes data and (correct) answers to produce the rules — the model.
      </Note>

      <H2 c="Core components of a learning system" />
      <Table
        heads={['Component', 'Role', 'Example']}
        rows={[
          ['Data',                  'Experience E the algorithm learns from', 'Labeled house prices with square footage, location'],
          ['Hypothesis class',      'The set of functions the model can represent', 'All linear functions y = wx + b'],
          ['Loss function',         'Measures how wrong a prediction is',      'Mean squared error, cross-entropy'],
          ['Optimization algorithm','Searches the hypothesis class to minimize loss', 'Gradient descent, closed-form least squares'],
        ]}
      />
      <Mx block>{`  Learning as optimization:
  θ* = argmin_θ  (1/n) Σᵢ L(f_θ(xᵢ), yᵢ)

  θ  = model parameters (weights)
  f_θ = the model, drawn from the hypothesis class
  L   = loss function comparing prediction f_θ(xᵢ) to true label yᵢ`}</Mx>

      <H2 c="Interview Q&A" />
      <QA q="What is Tom Mitchell's formal definition of machine learning?"
          a="A program learns from experience E with respect to task T and performance measure P if its performance on T, as measured by P, improves with E. It's useful because it forces you to name all three parts concretely — for a recommender system, T = predicting which items a user clicks, E = historical click logs, P = click-through rate on held-out users." />
      <QA q="How does ML differ fundamentally from traditional rule-based programming?"
          a="Traditional programming takes explicit rules plus data and produces answers. Machine learning takes data plus known answers (labels) and produces the rules — encoded as a model's parameters. This inversion is what lets ML systems handle tasks (like image recognition) where nobody can hand-write a complete rule set." />
      <QA q="What is a hypothesis space, and why does its size matter?"
          a="The hypothesis space (or hypothesis class) is the full set of candidate functions a learning algorithm is allowed to choose from — e.g. 'all linear functions' or 'all decision trees of depth ≤ 5'. A space that's too small can't represent the true pattern (underfitting / high bias). A space that's too large can fit noise in the training data (overfitting / high variance). Model selection is largely about choosing a hypothesis space of the right size for the amount of data available." />
    </div>
  );
}

/* ─────────────────── Section: Types of Learning ─────────────── */
function SectionTypes() {
  return (
    <div>
      <P>
        ML algorithms are usually grouped by the kind of feedback they learn from. The split isn't
        just academic — it determines what data you need to collect before you can train anything.
      </P>

      <H2 c="Supervised learning" />
      <P>
        Every training example has a known correct answer (a label). The model learns a mapping
        <Mx> x → y</Mx> and is graded against the true <Mx>y</Mx> at every step.
      </P>
      <Table
        heads={['', 'Regression', 'Classification']}
        rows={[
          ['Output type',  'Continuous number', 'Discrete category'],
          ['Example loss', 'Mean squared error', 'Cross-entropy / log loss'],
          ['Algorithms',   'Linear Regression, Ridge/Lasso, SVR, Gradient Boosting Regressor', 'Logistic Regression, k-NN, SVM, Decision Trees, Random Forest'],
          ['Example task', 'Predict house price from square footage', 'Classify an email as spam or not'],
        ]}
      />

      <H2 c="Unsupervised learning" />
      <P>There are no labels — the algorithm finds structure in the data on its own.</P>
      <Grid cols={3} gap={10}>
        <Card color="info" title="Clustering">
          Group similar points together.<br />
          <Badge color="info">K-Means</Badge>
          <Badge color="info">Hierarchical</Badge>
          <Badge color="info">DBSCAN</Badge>
        </Card>
        <Card color="success" title="Dimensionality Reduction">
          Compress features while keeping structure.<br />
          <Badge color="success">PCA</Badge>
          <Badge color="success">t-SNE</Badge>
          <Badge color="success">UMAP</Badge>
        </Card>
        <Card color="purple" title="Association Rules">
          Find items that co-occur.<br />
          <Badge color="purple">Apriori</Badge>
          <Badge color="purple">FP-Growth</Badge>
        </Card>
      </Grid>

      <H2 c="Semi-supervised & self-supervised learning" />
      <P>
        Semi-supervised learning trains on a small labeled set plus a much larger unlabeled set —
        useful when labels are expensive (e.g. medical imaging). Self-supervised learning generates
        labels automatically from the data itself (e.g. "predict the next word", "predict the masked
        patch") — it's the technique behind most modern pretrained language and vision models, so
        it's covered in more depth on the companion Deep Learning platform.
      </P>

      <H2 c="Reinforcement learning" />
      <P>
        An agent interacts with an environment, takes actions, and receives rewards. There's no
        "correct answer" per step — only a delayed signal of how good a sequence of decisions was.
      </P>
      <Mx block>{`  Agent–environment loop, at each timestep t:
    observe state sₜ → take action aₜ → receive reward rₜ, new state sₜ₊₁

  Objective: find a policy π that maximizes expected discounted return
    G_t = Σₖ γᵏ · r_{t+k}          (0 ≤ γ ≤ 1 is the discount factor)`}</Mx>

      <H2 c="Comparing the paradigms" />
      <Table
        heads={['Paradigm', 'Data', 'Goal', 'Example applications']}
        rows={[
          ['Supervised',     'Labeled (x, y) pairs',         'Predict y for new x',             'Fraud detection, price prediction'],
          ['Unsupervised',   'Unlabeled x only',              'Discover structure in x',          'Customer segmentation, anomaly detection'],
          ['Semi-supervised','Few labels + lots of unlabeled x', 'Leverage unlabeled data to help supervision', 'Medical imaging with scarce labels'],
          ['Reinforcement',  'State–action–reward sequences', 'Learn a reward-maximizing policy', 'Game playing, robotics, recommendation ranking'],
        ]}
      />

      <H3 c="A first taste of code" />
      <Code>{`from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)              # learn from experience E
print(clf.score(X_test, y_test))       # performance measure P`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="What's the core difference between supervised and unsupervised learning?"
          a="Supervised learning trains on labeled examples (x, y) and is evaluated by how close predictions ŷ come to the true y. Unsupervised learning only has x — there's no ground truth to check against, so the algorithm optimizes a proxy objective instead, like minimizing within-cluster distance (K-Means) or maximizing reconstructed variance (PCA)." />
      <QA q="When would you reach for semi-supervised learning over fully supervised?"
          a="When labeling is expensive or slow relative to collecting raw data — common in medical imaging, legal document review, or speech transcription. You label a small representative subset, then use techniques like pseudo-labeling or label propagation to extend supervision to the much larger unlabeled pool, often reaching accuracy close to a fully labeled dataset at a fraction of the labeling cost." />
      <QA q="How does reinforcement learning differ from supervised learning in terms of feedback?"
          a="Supervised learning gets an immediate, correct target for every training example. RL only gets a scalar reward, often delayed and dependent on a whole sequence of actions (credit assignment problem), and the agent's own actions determine what data it sees next (unlike i.i.d. training data in supervised learning)." />
    </div>
  );
}

/* ─────────────────── Section: The ML Workflow ───────────────── */
function SplitDemo() {
  const [trainPct, setTrainPct] = useState(70);
  const remaining = 100 - trainPct;
  const valPct = Math.round(remaining / 2);
  const testPct = remaining - valPct;
  const total = 1000;
  const trainN = Math.round((total * trainPct) / 100);
  const valN = Math.round((total * valPct) / 100);
  const testN = total - trainN - valN;

  const chip = (label, n, pct, color) => (
    <div style={{
      flex: pct, minWidth: 0, background: `var(--color-background-${color})`,
      color: `var(--color-text-${color})`, padding: '8px 6px', borderRadius: 6,
      textAlign: 'center', fontSize: 12,
    }}>
      {label}<br /><strong style={{ fontSize: 13.5 }}>{n}</strong>
    </div>
  );

  return (
    <VizBox>
      <SliderRow
        label="Train %"
        min={50} max={90} step={5}
        value={trainPct}
        onChange={setTrainPct}
        fmt={v => `${v}%`}
      />
      <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
        {chip('Train', trainN, trainPct, 'info')}
        {chip('Val', valN, valPct, 'warning')}
        {chip('Test', testN, testPct, 'success')}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Splitting {total} samples — remaining {remaining}% after training is split evenly between validation and test.
      </div>
    </VizBox>
  );
}

function SectionWorkflow() {
  const stages = [
    { n: '01', stage: 'Problem framing',       desc: 'Translate a business question into a learning task — regression? classification? clustering?' },
    { n: '02', stage: 'Data collection',       desc: 'Gather data that actually represents the distribution you’ll see in production.' },
    { n: '03', stage: 'Exploratory analysis',  desc: 'Understand distributions, missing values, correlations, and outliers before touching a model.' },
    { n: '04', stage: 'Cleaning & preprocessing', desc: 'Handle missing values, encode categoricals, scale numeric features.' },
    { n: '05', stage: 'Feature engineering',   desc: 'Construct informative inputs — often matters more than the choice of algorithm.' },
    { n: '06', stage: 'Model selection',       desc: 'Pick a hypothesis class suited to the data size, structure, and interpretability needs.' },
    { n: '07', stage: 'Training',              desc: 'Fit model parameters by minimizing a loss function on the training split.' },
    { n: '08', stage: 'Evaluation',            desc: 'Score the model on held-out data with a metric that matches the business goal.' },
    { n: '09', stage: 'Hyperparameter tuning', desc: 'Search configurations (grid/random/Bayesian) to improve validation performance.' },
    { n: '10', stage: 'Deployment',            desc: 'Serve the trained model behind an API, batch job, or embedded runtime.' },
    { n: '11', stage: 'Monitoring & retraining', desc: 'Track data/label drift in production and retrain as the world changes.' },
  ];

  return (
    <div>
      <P>Every production ML system moves through the same rough lifecycle, whether it's a weekend project or a team's flagship model.</P>

      <H2 c="The end-to-end ML lifecycle" />
      <div style={{ position: 'relative', paddingLeft: 24, borderLeft: '2px solid var(--color-border-secondary)', margin: '1rem 0' }}>
        {stages.map(({ n, stage, desc }) => (
          <div key={n} style={{ marginBottom: 12, position: 'relative' }}>
            <div style={{ position: 'absolute', left: -30, top: 3, width: 12, height: 12, borderRadius: '50%', background: 'var(--color-background-info)', border: '2px solid var(--color-border-info)' }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 500, color: 'var(--color-text-info)' }}>{n}</span>
            <span style={{ fontSize: 13, color: 'var(--color-text-primary)', marginLeft: 10, fontWeight: 500 }}>{stage}</span>
            <div style={{ fontSize: 12.5, color: 'var(--color-text-secondary)', marginLeft: 34, marginTop: 2 }}>{desc}</div>
          </div>
        ))}
      </div>

      <H2 c="Train / validation / test split" />
      <P>
        The training set fits parameters. The validation set tunes hyperparameters and model choice.
        The test set is touched exactly once, at the very end, to estimate real-world performance.
      </P>
      <Mx block>{`  Common splits:
    Small/medium data (≤100K rows):  60/20/20  or  70/15/15
    Very large data (millions+):     98/1/1  is often enough —
      1% of 10M rows is still 100K validation examples`}</Mx>
      <Note color="warning" icon="ti-alert-triangle">
        Data leakage: fitting a scaler, imputer, or feature selector on the <em>full</em> dataset before
        splitting lets test-set statistics leak into training. Always split first, then fit
        preprocessing only on the training fold.
      </Note>

      <H3 c="Try it — split ratio" />
      <SplitDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1-2. Data collection (built-in dataset stands in for a real source)
X, y = load_breast_cancer(return_X_y=True)

# 3-4. Split before touching any statistics — avoids leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5-7. Scaling + model composed as one fit/predict unit
pipeline = Pipeline([
    ("scaler", StandardScaler()),      # fit only on X_train internally
    ("clf", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)

# 8. Evaluation on held-out data
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why do we need a separate validation set in addition to a test set?"
          a="If you tune hyperparameters or pick a model by repeatedly checking performance on the test set, you're indirectly fitting to the test set — its score stops being an honest estimate of generalization. The validation set absorbs all that iterative decision-making; the test set is touched once, at the end, so its score stays a trustworthy estimate of real-world performance." />
      <QA q="What is data leakage, and how can it sneak in during preprocessing?"
          a="Data leakage is when information from outside the training set (often the test set, or the future) influences training, producing metrics that look great but don't hold up in production. A classic case: computing a StandardScaler's mean/std on the full dataset before splitting — the training fold then 'knows' something about the test fold's distribution. Fitting all preprocessing exclusively on the training fold (e.g. inside a Pipeline or cross-validation loop) prevents this." />
      <QA q="What's the difference between training error and generalization error?"
          a="Training error is measured on the exact data the model was fit on. Generalization error is the expected error on new, unseen data drawn from the same distribution — approximated in practice by the test error. A large gap where training error is low but generalization error is high is the signature of overfitting." />
    </div>
  );
}

/* ─────────────────── Section: ML vs Deep Learning ───────────── */
function SectionMLvDL() {
  return (
    <div>
      <P>
        Deep Learning is a subset of Machine Learning — every DL model is an ML model, but not
        every ML model is deep. The practical dividing line is usually how features get made.
      </P>

      <H2 c="Detailed comparison" />
      <Table
        heads={['Dimension', 'Classical ML', 'Deep Learning']}
        rows={[
          ['Feature engineering',      'Manual, domain expertise required', 'Automatic, learned end-to-end from raw data'],
          ['Data requirements',        'Works well on small/medium (~1K–100K rows)', 'Typically needs large datasets (100K–millions+)'],
          ['Compute',                  'CPU, seconds–minutes',           'GPU/TPU, hours–weeks'],
          ['Interpretability',         'Often high (linear models, trees)', 'Usually low — treated as a "black box"'],
          ['Performance on tabular',   'Frequently wins (XGBoost, LightGBM)', 'Comparable or worse without heavy tuning'],
          ['Performance on unstructured', 'Requires hand-crafted features', 'State-of-the-art (images, text, audio)'],
          ['Typical algorithms',       'Linear/Logistic Regression, Trees, SVM, k-NN, Naive Bayes, Gradient Boosting', 'CNNs, RNNs, Transformers, GANs'],
        ]}
      />

      <H2 c="When to use each" />
      <Grid cols={2} gap={12}>
        <Card color="success" title="Use classical ML when…">
          Data &lt; 100K rows · Structured/tabular data · Need interpretability · Limited compute · Fast iteration required
        </Card>
        <Card color="purple" title="Use deep learning when…">
          Images, text, or audio · Raw features are hard to hand-engineer · GPU available · A large labeled (or pretrainable) dataset exists
        </Card>
      </Grid>

      <Note color="info" icon="ti-external-link">
        This platform covers classical ML end-to-end. For neural-network architectures — CNNs, RNNs,
        Transformers — see the companion{' '}
        <a href="https://jitamb.github.io/Deep-Learning-Study-Platform/" target="_blank" rel="noreferrer" style={{ textDecoration: 'underline' }}>
          Deep Learning Study Guide
        </a>.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why does gradient boosting (e.g. XGBoost) often outperform deep learning on tabular data?"
          a="Tabular datasets tend to be small relative to image/text corpora, mix numeric and categorical features, and contain irregular, non-smooth decision boundaries that tree splits capture naturally. Tree ensembles handle missing values and categorical variables with little preprocessing, and don't need the huge sample sizes deep networks rely on to learn good representations. Multiple benchmark studies (e.g. Grinsztajn et al., 2022) confirm tree-based models still lead on typical tabular benchmarks." />
      <QA q="Is deep learning a strict subset of machine learning, or something separate?"
          a="A strict subset. Machine learning is the general discipline of learning patterns from data; deep learning specifically means doing so with multi-layer (‘deep’) neural networks that learn hierarchical feature representations automatically. Every deep learning model is trained using ML principles (loss functions, optimization, train/test splits) — it's a family of models and techniques within ML, not a separate field." />
    </div>
  );
}

/* ─────────────────── Page Root ─────────────────────────────── */
const SECTION_MAP = {
  def:      <SectionDef />,
  types:    <SectionTypes />,
  workflow: <SectionWorkflow />,
  mldl:     <SectionMLvDL />,
};

export default function MLFundamentals() {
  const [active, setActive] = useState('def');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 01</div>
        <h1 className="page-header-title">ML Fundamentals</h1>
        <p className="page-header-subtitle">What machine learning is, the major learning paradigms, and the workflow that turns data into a deployed model.</p>
      </div>

      <SectionNav tabs={TABS} active={active} onSelect={setActive} />

      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>

      <NavButtons moduleId={1} />
    </div>
  );
}
