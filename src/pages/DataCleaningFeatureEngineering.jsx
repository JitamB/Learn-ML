import { useState } from 'react';
import { Mx, H2, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

/* ── Shared helpers ─────────────────────────────────────────── */
function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function mode(arr) {
  const counts = {};
  arr.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
  return Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0]);
}
function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 14px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}

/* ── Missing Data: imputation demo ─────────────────────────── */
const IMPUTE_COLUMN = [42, 38, null, 45, 51, null, 40];

function ImputationDemo() {
  const [strategy, setStrategy] = useState('mean');
  const known = IMPUTE_COLUMN.filter(v => v !== null);
  const meanVal = known.reduce((a, b) => a + b, 0) / known.length;
  const fillValue = strategy === 'mean' ? meanVal : strategy === 'median' ? median(known) : mode(known);

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        {['mean', 'median', 'mode'].map(s => (
          <button key={s} onClick={() => setStrategy(s)} style={{ ...toggleBtnStyle(strategy === s), textTransform: 'capitalize' }}>
            {s}
          </button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {IMPUTE_COLUMN.map((v, i) => (
          <div key={i} style={{
            width: 50, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center',
            borderRadius: 'var(--border-radius-md)', fontSize: 12.5, fontFamily: 'var(--font-mono)',
            border: '1px solid ' + (v === null ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
            background: v === null ? 'var(--color-background-info)' : 'transparent',
            color: v === null ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
            fontWeight: v === null ? 600 : 400,
          }}>
            {v === null ? fillValue.toFixed(1) : v}
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        The 2 highlighted cells were missing — both filled with the column's {strategy} ({fillValue.toFixed(1)}).
      </div>
    </VizBox>
  );
}

/* ── Outliers: threshold strip demo ────────────────────────── */
const OUTLIER_VALUES = [47, 49, 50, 52, 48, 51, 53, 46, 50, 49, 52, 48, 51, 50, 65, 25, 85];

function OutlierStrip() {
  const [threshold, setThreshold] = useState(2.5);
  const meanVal = OUTLIER_VALUES.reduce((a, b) => a + b, 0) / OUTLIER_VALUES.length;
  const std = Math.sqrt(OUTLIER_VALUES.reduce((a, b) => a + (b - meanVal) ** 2, 0) / OUTLIER_VALUES.length);
  const zScores = OUTLIER_VALUES.map(v => (v - meanVal) / std);
  const min = Math.min(...OUTLIER_VALUES), max = Math.max(...OUTLIER_VALUES);
  const flaggedCount = zScores.filter(z => Math.abs(z) > threshold).length;

  return (
    <VizBox>
      <SliderRow label="Z threshold" min={1} max={4} step={0.25} value={threshold} onChange={setThreshold} fmt={v => `±${v.toFixed(2)}`} />
      <div style={{ position: 'relative', height: 50, marginTop: 18 }}>
        <div style={{ position: 'absolute', left: 0, right: 0, top: 22, height: 1, background: 'var(--color-border-tertiary)' }} />
        {OUTLIER_VALUES.map((v, i) => {
          const pct = ((v - min) / (max - min)) * 100;
          const flagged = Math.abs(zScores[i]) > threshold;
          return (
            <div key={i} title={`value ${v}, z=${zScores[i].toFixed(2)}`} style={{
              position: 'absolute', left: `${pct}%`, top: 14, width: 12, height: 12, borderRadius: '50%',
              background: flagged ? 'var(--color-text-danger)' : 'var(--color-border-info)',
              transform: 'translate(-50%, 0)', transition: 'background 0.15s',
            }} />
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 12 }}>
        {flaggedCount} of {OUTLIER_VALUES.length} points flagged at ±{threshold.toFixed(2)}σ. Lower the
        threshold and ordinary-looking points start getting caught too.
      </div>
    </VizBox>
  );
}

/* ── Feature Engineering: binning demo ─────────────────────── */
const BINNING_VALUES = [12, 19, 23, 28, 31, 35, 42, 47, 53, 58, 65, 71];
const BIN_PALETTE = ['info', 'success', 'warning', 'danger', 'purple'];

function BinningDemo() {
  const [numBins, setNumBins] = useState(4);
  const min = Math.min(...BINNING_VALUES), max = Math.max(...BINNING_VALUES);
  const width = (max - min) / numBins;
  const binOf = v => Math.min(numBins - 1, Math.floor((v - min) / width));

  return (
    <VizBox>
      <SliderRow label="Bins" min={2} max={8} step={1} value={numBins} onChange={setNumBins} fmt={v => `${v}`} />
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 14 }}>
        {BINNING_VALUES.map((v, i) => {
          const color = BIN_PALETTE[binOf(v) % BIN_PALETTE.length];
          return (
            <div key={i} style={{
              padding: '5px 10px', borderRadius: 'var(--border-radius-md)', fontSize: 12, fontFamily: 'var(--font-mono)',
              background: `var(--color-background-${color})`, color: `var(--color-text-${color})`,
              border: `1px solid var(--color-border-${color})`,
            }}>{v}</div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Same {BINNING_VALUES.length} values, split into {numBins} equal-width bins — same-colored chips
        fall in the same bin.
      </div>
    </VizBox>
  );
}

/* ── Encoding & Scaling: scaler comparison demo ────────────── */
const SCALE_VALUES = [12, 15, 14, 13, 16, 95, 14, 15];

function standardize(arr) {
  const meanVal = arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = Math.sqrt(arr.reduce((a, b) => a + (b - meanVal) ** 2, 0) / arr.length);
  return arr.map(v => (v - meanVal) / std);
}
function minMaxScale(arr) {
  const min = Math.min(...arr), max = Math.max(...arr);
  return arr.map(v => (v - min) / (max - min));
}
function robustScale(arr) {
  const med = median(arr);
  const sorted = [...arr].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1 || 1;
  return arr.map(v => (v - med) / iqr);
}

function ScalerDemo() {
  const [scaler, setScaler] = useState('standard');
  const scaled = scaler === 'standard' ? standardize(SCALE_VALUES) : scaler === 'minmax' ? minMaxScale(SCALE_VALUES) : robustScale(SCALE_VALUES);
  const outlierValue = Math.max(...SCALE_VALUES);

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        {[{ id: 'standard', label: 'Standard' }, { id: 'minmax', label: 'Min-Max' }, { id: 'robust', label: 'Robust' }].map(s => (
          <button key={s.id} onClick={() => setScaler(s.id)} style={toggleBtnStyle(scaler === s.id)}>{s.label}</button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {SCALE_VALUES.map((v, i) => {
          const isOutlier = v === outlierValue;
          return (
            <div key={i} style={{
              padding: '5px 10px', borderRadius: 'var(--border-radius-md)', fontSize: 12, fontFamily: 'var(--font-mono)', textAlign: 'center', minWidth: 54,
              border: '1px solid ' + (isOutlier ? 'var(--color-border-danger)' : 'var(--color-border-tertiary)'),
              color: isOutlier ? 'var(--color-text-danger)' : 'var(--color-text-secondary)',
            }}>
              <div style={{ fontSize: 10.5, opacity: 0.7 }}>{v}</div>
              <div style={{ fontWeight: 600 }}>{scaled[i].toFixed(2)}</div>
            </div>
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        {scaler === 'minmax'
          ? 'The one outlier (95) stretches the whole 0–1 range, crushing every other point toward 0.'
          : scaler === 'standard'
          ? 'The outlier inflates the mean and std, pulling even the non-outliers toward 0.'
          : 'Median and IQR barely notice the outlier — the other points keep sensible, spread-out values.'}
      </div>
    </VizBox>
  );
}

/* ── Imbalanced Data: class-balance demo ───────────────────── */
function ImbalanceDemo() {
  const [minorityPct, setMinorityPct] = useState(10);
  const total = 200;
  const minorityCount = Math.round((minorityPct / 100) * total);
  const majorityCount = total - minorityCount;
  const dumbAccuracy = (majorityCount / total) * 100;

  return (
    <VizBox>
      <SliderRow label="Minority %" min={1} max={50} step={1} value={minorityPct} onChange={setMinorityPct} fmt={v => `${v}%`} />
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 16, height: 90, marginTop: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 48, height: `${(majorityCount / total) * 70}px`, background: 'var(--color-border-info)', borderRadius: '4px 4px 0 0' }} />
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{majorityCount} majority</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 48, height: `${Math.max(2, (minorityCount / total) * 70)}px`, background: 'var(--color-text-danger)', borderRadius: '4px 4px 0 0' }} />
          <span style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>{minorityCount} minority</span>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: 22, fontWeight: 600, color: 'var(--color-text-primary)', fontFamily: 'var(--font-mono)' }}>{dumbAccuracy.toFixed(1)}%</div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)' }}>accuracy from always guessing "majority"</div>
        </div>
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'missing', label: 'Missing Data', sub: 'Mechanisms & imputation' },
  { id: 'outliers', label: 'Outliers', sub: 'Detection & treatment' },
  { id: 'feature-eng', label: 'Feature Engineering & Selection', sub: 'Creating & choosing features' },
  { id: 'encoding-scaling', label: 'Encoding & Scaling', sub: 'Categorical & numeric prep' },
  { id: 'imbalanced', label: 'Imbalanced Data', sub: 'When one class dominates' },
];

function SectionMissing() {
  return (
    <div>
      <P>
        Almost every real dataset has gaps. What you do about them depends entirely on <em>why</em>{' '}
        they're there — the mechanism decides which fixes are safe.
      </P>

      <H2 c="Why Is It Missing?" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="MCAR">
          Missing Completely At Random — the gap has nothing to do with any value, observed or not.
          Safest case; dropping rows doesn't bias the result.
        </Card>
        <Card color="warning" title="MAR">
          Missing At Random — the chance of being missing depends on other observed columns (e.g.
          income missing more often for younger respondents).
        </Card>
        <Card color="danger" title="MNAR">
          Missing Not At Random — the chance of being missing depends on the missing value itself
          (e.g. high earners skipping the income question). Deletion and simple imputation both bias
          the result.
        </Card>
      </Grid>

      <H2 c="Deletion" />
      <P>
        Safe only under MCAR, and only when the missing fraction is small. <strong>Listwise
        deletion</strong> drops any row with a missing value; dropping a whole <strong>column</strong>{' '}
        is the better call once most of its values are missing.
      </P>

      <H2 c="Imputation" />
      <Table
        heads={['Method', 'Fills with', 'Trade-off']}
        rows={[
          ['Mean / Median', 'The column average (median if skewed or outlier-heavy)', 'Fast, but shrinks variance and ignores relationships to other columns'],
          ['Mode / Constant', 'The most frequent category, or a fixed placeholder like "Unknown"', 'Standard for categoricals; a placeholder can itself become a signal if missingness is meaningful'],
          ['KNN Imputer', 'The average of the k most similar rows (by the other columns)', 'Uses relationships between columns; slower, sensitive to feature scaling'],
          ['Iterative Imputer (MICE)', 'A regression model predicting each missing value from the rest of the row', 'Most accurate for MAR data; the most expensive to compute'],
        ]}
      />

      <Code>{`from sklearn.impute import KNNImputer

df["age"] = df["age"].fillna(df["age"].median())

imputer = KNNImputer(n_neighbors=5)
df[["age", "income"]] = imputer.fit_transform(df[["age", "income"]])`}</Code>

      <H2 c="Fill the Gaps" />
      <P>Two missing cells in this column — watch the fill value change with the strategy.</P>
      <ImputationDemo />

      <H2 c="Interview Q&A" />
      <QA
        q="Why does the missingness mechanism (MCAR/MAR/MNAR) matter more than just the percentage missing?"
        a="It determines whether a fix introduces bias. Under MCAR, dropping rows or imputing with the mean leaves the remaining data representative; under MNAR, the very fact that a value is missing carries information, so both deletion and naive imputation systematically distort the result."
      />
      <QA
        q="When would you reach for median imputation instead of mean?"
        a="When the column is skewed or has outliers — the mean gets pulled by extreme values, while the median stays representative of a 'typical' row."
      />
      <QA
        q="What's the actual downside of mean/median imputation beyond 'it's not perfectly accurate'?"
        a="It artificially shrinks the column's variance (every imputed row gets the exact same value) and erases whatever correlation that column had with others, which can understate a model's real uncertainty."
      />
    </div>
  );
}

function SectionOutliers() {
  return (
    <div>
      <P>
        An outlier isn't automatically an error — it might be the most important row in the dataset
        (fraud, a rare failure mode). Detection just finds candidates; deciding what they mean is a
        separate step.
      </P>

      <H2 c="Statistical Detection" />
      <Mx block>{`  Z-score:

    Z = (x - μ) / σ

  μ = column mean, σ = column standard deviation.
  |Z| > 3 is a common (not universal) cutoff.`}</Mx>
      <Mx block>{`  IQR method:

    IQR = Q3 - Q1
    lower bound = Q1 - 1.5 × IQR
    upper bound = Q3 + 1.5 × IQR

  Anything outside [lower bound, upper bound] is flagged.`}</Mx>

      <H2 c="Algorithmic Detection" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Isolation Forest">
          Randomly splits the data repeatedly; outliers need fewer splits to isolate than typical
          points do. Scales well, works in high dimensions.
        </Card>
        <Card color="success" title="Local Outlier Factor (LOF)">
          Compares a point's local density to its neighbors' — flags points sitting in a sparser
          region than their surroundings, catching outliers a single global threshold would miss.
        </Card>
      </Grid>

      <H2 c="Treatment" />
      <Table
        heads={['Strategy', 'What it does', 'When to use it']}
        rows={[
          ['Trimming', 'Removes flagged rows entirely', 'A handful of clear data-entry errors'],
          ['Winsorizing', 'Caps values at a percentile (e.g. clip to the 1st/99th) instead of removing them', 'Keeping the row but limiting the extreme value’s influence'],
          ['Transformation', 'Log or other transform that compresses the scale of extreme values', 'Outliers are real but the model is sensitive to raw scale (e.g. linear models)'],
        ]}
      />

      <Code>{`from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.02, random_state=42)
df["is_outlier"] = iso.fit_predict(df[["amount"]]) == -1

lower, upper = df["amount"].quantile([0.01, 0.99])
df["amount_winsorized"] = df["amount"].clip(lower, upper)`}</Code>

      <H2 c="Threshold, Visualized" />
      <P>Same 17 points — drag the Z-score threshold and watch which ones get flagged.</P>
      <OutlierStrip />

      <H2 c="Interview Q&A" />
      <QA
        q="Why would you use Isolation Forest instead of a simple Z-score cutoff?"
        a="Z-score only looks at one column at a time and assumes a roughly normal distribution; Isolation Forest handles multiple columns jointly and doesn't assume any particular distribution, catching multivariate outliers that look normal on every individual column but unusual in combination."
      />
      <QA
        q="When is winsorizing a better choice than just dropping outlier rows?"
        a="When the row's other columns still carry useful information you don't want to lose — capping the extreme value keeps the row while limiting how much it can distort a mean, a distance calculation, or a model's loss."
      />
    </div>
  );
}

function SectionFeatureEngSelection() {
  return (
    <div>
      <P>
        Feature engineering creates candidate columns; feature selection decides which of them
        actually earn a place in the final model. Both exist because a model is only as good as the
        columns it's given.
      </P>

      <H2 c="Creating Features" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Polynomial Features">
          Adds interaction and power terms so a linear model can capture non-linear relationships it
          otherwise couldn't.
        </Card>
        <Card color="success" title="Binning / Discretization">
          Converts a continuous variable into ranges (e.g. age → 0-18, 19-35, 36-60, 60+) — trades
          precision for robustness to noise and outliers.
        </Card>
        <Card color="purple" title="Mathematical Transforms">
          Log, Box-Cox, or Yeo-Johnson pull in a long tail to make a skewed distribution closer to
          normal.
        </Card>
        <Card color="warning" title="Date/Time Extraction">
          Pulls actionable, often cyclical features — day of week, month, is-holiday — out of a raw
          timestamp string.
        </Card>
      </Grid>

      <Code>{`from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(df[["sqft", "age"]])

df["log_price"] = np.log1p(df["price"])
df["signup_dow"] = pd.to_datetime(df["signup"]).dt.dayofweek`}</Code>

      <H2 c="Bin It Yourself" />
      <P>Same 12 values, split into an adjustable number of equal-width bins — same color, same bin.</P>
      <BinningDemo />

      <H2 c="Selecting Features" />
      <Table
        heads={['Approach', 'How it works', 'Example methods']}
        rows={[
          ['Filter', 'Scores each feature against the target independently of any model', 'Pearson correlation, Chi-Square, ANOVA F-test'],
          ['Wrapper', 'Trains the actual model repeatedly on different feature subsets to find the best one', 'Forward Selection, Backward Elimination, Recursive Feature Elimination'],
          ['Embedded', 'Selection happens as a side effect of training a single model', 'Lasso (L1) coefficients shrink to exactly zero, tree feature importances'],
        ]}
      />

      <Note color="info" icon="ti-info-circle">
        Filter methods are the cheapest first pass on a wide dataset; wrapper methods are the most
        accurate but the most expensive since they retrain a model per candidate subset.
      </Note>

      <H2 c="Interview Q&A" />
      <QA
        q="Why would adding polynomial features ever hurt a model?"
        a="Every interaction and power term is a new column, so degree-2+ expansion across many features explodes the feature count — with enough of them the model can overfit, and multicollinearity between the new terms and the originals can make coefficients unstable."
      />
      <QA
        q="What's the practical difference between filter and wrapper feature selection?"
        a="Filter methods score each feature independently and never touch the actual model, so they're fast but ignore feature interactions; wrapper methods retrain the real model on different subsets, capturing interactions at far higher compute cost — filtering first, then wrapping on what's left, is a common compromise."
      />
    </div>
  );
}

function SectionEncodingScaling() {
  return (
    <div>
      <P>
        The last step before most models can even accept the data — turning categories into numbers,
        and putting numeric columns on comparable scales.
      </P>

      <H2 c="Categorical Encoding" />
      <Table
        heads={['Technique', 'Description', 'Best use case']}
        rows={[
          ['One-Hot Encoding', 'Creates a separate binary (0/1) column for each unique category', 'Nominal data with a low number of unique categories'],
          ['Label Encoding', 'Assigns each category an integer, in no particular order', 'A model that won’t misread the integers as having magnitude (e.g. tree-based models)'],
          ['Target Encoding', 'Replaces each category with the mean of the target variable for that category', 'High-cardinality nominal features (zip codes, user IDs)'],
          ['Frequency Encoding', 'Replaces each category with its overall count in the dataset', 'When how common a category is correlates with the target'],
        ]}
      />
      <Note color="warning" icon="ti-alert-triangle">
        Label Encoding assigns integers arbitrarily — it doesn't actually respect order. For a
        genuinely ordinal column (e.g. low/medium/high), map the categories to integers yourself in
        the correct order instead of letting a generic label encoder pick one.
      </Note>

      <H2 c="Feature Scaling" />
      <Table
        heads={['Scaler', 'Description', 'Best use case']}
        rows={[
          ['Standardization', 'Transforms to mean 0, standard deviation 1', 'Algorithms assuming roughly-normal inputs (linear models, SVMs, PCA)'],
          ['Min-Max Scaling', 'Scales to a fixed range, typically [0, 1]', 'Neural networks and anything sensitive to small variances'],
          ['Robust Scaling', 'Centers and scales using the median and IQR', 'Datasets with significant, untreated outliers'],
        ]}
      />

      <Code>{`from sklearn.preprocessing import StandardScaler

df = pd.get_dummies(df, columns=["city"])                    # one-hot
df["size_rank"] = df["size"].map({"S": 0, "M": 1, "L": 2})    # ordinal, mapped by hand

X_scaled = StandardScaler().fit_transform(df[["age", "income"]])`}</Code>

      <H2 c="One Outlier, Three Scalers" />
      <P>Same 8 values (one of them 95) — pick a scaler and watch what happens to it and its neighbors.</P>
      <ScalerDemo />

      <H2 c="Interview Q&A" />
      <QA
        q="Why can Label Encoding be dangerous for a linear model even though it's fine for a tree-based one?"
        a="A linear model treats the encoded integers as having real magnitude and order — encoding {red: 0, blue: 1, green: 2} implies green is 'twice' blue, which is meaningless for nominal data. Tree-based models only ever split on thresholds, so the arbitrary ordering doesn't mislead them the same way."
      />
      <QA
        q="A feature has a few extreme outliers you haven't removed yet. Which scaler should you reach for?"
        a="Robust Scaling — it centers and scales using the median and IQR, both of which barely move when a few extreme values are present, unlike Standardization (mean/std) or Min-Max (min/max), which the outliers directly distort."
      />
    </div>
  );
}

function SectionImbalanced() {
  return (
    <div>
      <P>
        When one class vastly outnumbers the other, a model can reach high accuracy by barely trying
        — which makes accuracy alone a dangerous metric to trust here.
      </P>

      <H2 c="Rebalancing the Data" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Oversampling">
          Duplicates minority-class examples, or generates synthetic ones with SMOTE (Synthetic
          Minority Over-sampling Technique) by interpolating between real minority points.
        </Card>
        <Card color="success" title="Undersampling">
          Randomly removes majority-class examples, or uses a targeted method like Tomek Links to
          remove majority points sitting right on the decision boundary.
        </Card>
      </Grid>

      <H2 c="Rebalancing the Model Instead" />
      <P>
        Rather than touching the data, <strong>class-weight</strong> parameters tell the model itself
        to penalize misclassifying the minority class more heavily during training — no synthetic
        data, no discarded rows.
      </P>

      <Code>{`from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

clf = LogisticRegression(class_weight="balanced")
clf.fit(X_train, y_train)`}</Code>

      <H2 c="Why Accuracy Lies Here" />
      <P>Drag the minority share down — watch a model that predicts nothing but "majority" look better and better.</P>
      <ImbalanceDemo />

      <Note color="danger" icon="ti-alert-triangle">
        A classifier that always predicts the majority class needs zero learning to hit that accuracy
        number. How to actually measure a model fairly under imbalance — precision, recall, F1,
        ROC-AUC — is covered in Metrics under Model Evaluation & Validation.
      </Note>

      <H2 c="Interview Q&A" />
      <QA
        q="Why is accuracy a misleading metric on an imbalanced dataset?"
        a="A model that ignores the minority class entirely and always predicts the majority class still scores an accuracy equal to the majority class's share of the data — 99% imbalance means 99% 'accuracy' from a model that has learned nothing useful about the minority class."
      />
      <QA
        q="When would you reach for class-weight adjustment instead of SMOTE?"
        a="Class-weight avoids fabricating synthetic data or discarding real rows, so it's usually the first thing to try — SMOTE becomes more useful once the minority class is so small that no amount of reweighting gives the model enough signal to learn its pattern."
      />
    </div>
  );
}

const SECTION_MAP = {
  missing: <SectionMissing />,
  outliers: <SectionOutliers />,
  'feature-eng': <SectionFeatureEngSelection />,
  'encoding-scaling': <SectionEncodingScaling />,
  imbalanced: <SectionImbalanced />,
};

export default function DataCleaningFeatureEngineering() {
  const [active, setActive] = useState('missing');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 03</div>
        <h1 className="page-header-title">Data Cleaning & Feature Engineering</h1>
        <p className="page-header-subtitle">
          Turning raw, messy data into model-ready features. Almost every later module assumes the
          data it receives has already been through this stage.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={3} />
    </div>
  );
}
