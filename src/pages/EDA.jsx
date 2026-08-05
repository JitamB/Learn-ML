import { useState } from 'react';
import { Mx, H2, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

/* ── Data Sanity Checks demo ───────────────────────────────── */
const SANITY_ROWS = [
  { id: 101, age: 34, city: 'Pune', signup: '2023-01-14' },
  { id: 102, age: null, city: 'Delhi', signup: '2023-02-02' },
  { id: 103, age: 29, city: 'Bengaluru', signup: '2023-01-30' },
  { id: 104, age: 41, city: null, signup: '2023-03-19' },
  { id: 105, age: 34, city: 'Pune', signup: '2023-01-14' },
  { id: 106, age: 52, city: 'Mumbai', signup: '2023-04-02' },
];
const SANITY_COLS = ['id', 'age', 'city', 'signup'];
const SANITY_MODES = [
  { id: 'duplicates', label: 'Highlight Duplicates' },
  { id: 'missing', label: 'Highlight Missing' },
  { id: 'cardinality', label: 'Highlight High-Cardinality' },
];

function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 12px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}

function SanityCheckDemo() {
  const [mode, setMode] = useState('duplicates');
  const dupKey = r => `${r.age}|${r.city}|${r.signup}`;
  const dupCounts = SANITY_ROWS.reduce((acc, r) => { const k = dupKey(r); acc[k] = (acc[k] || 0) + 1; return acc; }, {});
  const isDupRow = r => dupCounts[dupKey(r)] > 1;
  const isHighlightedCell = (r, col) => {
    if (mode === 'missing') return r[col] === null;
    if (mode === 'cardinality') return col === 'id';
    return false;
  };

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        {SANITY_MODES.map(btn => (
          <button key={btn.id} onClick={() => setMode(btn.id)} style={toggleBtnStyle(mode === btn.id)}>
            {btn.label}
          </button>
        ))}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
          <thead>
            <tr>
              {SANITY_COLS.map(c => (
                <th key={c} style={{
                  textAlign: 'left', padding: '6px 10px', borderBottom: '1px solid var(--color-border-tertiary)',
                  fontWeight: 500,
                  color: mode === 'cardinality' && c === 'id' ? 'var(--color-text-danger)' : 'var(--color-text-primary)',
                }}>{c}{mode === 'cardinality' && c === 'id' ? ' ⚠' : ''}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {SANITY_ROWS.map((r, i) => (
              <tr key={i} style={{ background: mode === 'duplicates' && isDupRow(r) ? 'var(--color-background-danger)' : 'transparent' }}>
                {SANITY_COLS.map(c => (
                  <td key={c} style={{
                    padding: '6px 10px', borderBottom: '1px solid var(--color-border-tertiary)',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: isHighlightedCell(r, c) ? 600 : 400,
                    color: isHighlightedCell(r, c) ? 'var(--color-text-danger)' : 'var(--color-text-secondary)',
                  }}>
                    {r[c] === null ? (mode === 'missing' ? 'NaN' : '—') : r[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        {mode === 'duplicates' && 'Rows 1 and 5 share the same age/city/signup — likely the same record entered twice.'}
        {mode === 'missing' && 'age is missing in row 2, city is missing in row 4.'}
        {mode === 'cardinality' && 'id is unique on every row — a strong signal it identifies records rather than predicting anything.'}
      </div>
    </VizBox>
  );
}

/* ── Skew histogram demo ───────────────────────────────────── */
function skewedBarHeight(i, skew, peak, n) {
  const spreadLeft = 3 + Math.max(0, -skew) * 4;
  const spreadRight = 3 + Math.max(0, skew) * 6;
  const spread = i < peak ? spreadLeft : spreadRight;
  const z = (i - peak) / spread;
  return Math.exp(-0.5 * z * z);
}

function SkewHistogram() {
  const [skew, setSkew] = useState(0.5);
  const n = 20;
  const peak = 6;
  const heights = Array.from({ length: n }, (_, i) => skewedBarHeight(i, skew, peak, n));
  const maxH = Math.max(...heights);
  const norm = heights.map(h => h / maxH);
  const total = heights.reduce((a, b) => a + b, 0);
  const mean = heights.reduce((sum, h, i) => sum + h * i, 0) / total;
  let cum = 0, median = 0;
  for (let i = 0; i < n; i++) { cum += heights[i]; if (cum / total >= 0.5) { median = i; break; } }

  return (
    <VizBox>
      <SliderRow label="Skew" min={-1} max={1} step={0.1} value={skew} onChange={setSkew} fmt={v => v.toFixed(1)} />
      <div style={{ position: 'relative', height: 140, display: 'flex', alignItems: 'flex-end', gap: 2, marginTop: 14, borderBottom: '1px solid var(--color-border-tertiary)', paddingBottom: 2 }}>
        {norm.map((h, i) => (
          <div key={i} style={{ flex: 1, height: `${Math.max(2, h * 100)}%`, background: 'var(--color-border-info)', borderRadius: '2px 2px 0 0', opacity: 0.85 }} />
        ))}
        <div style={{ position: 'absolute', left: `${((mean + 0.5) / n) * 100}%`, top: 0, bottom: 0, width: 2, background: 'var(--color-text-danger)' }} title="Mean" />
        <div style={{ position: 'absolute', left: `${((median + 0.5) / n) * 100}%`, top: 0, bottom: 0, width: 2, background: 'var(--color-text-primary)' }} title="Median" />
      </div>
      <div style={{ display: 'flex', gap: 16, fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8, flexWrap: 'wrap' }}>
        <span><span style={{ color: 'var(--color-text-danger)' }}>■</span> Mean</span>
        <span><span style={{ color: 'var(--color-text-primary)' }}>■</span> Median</span>
        <span style={{ marginLeft: 'auto' }}>
          {Math.abs(skew) < 0.15 ? 'Symmetric — mean ≈ median' : skew > 0 ? 'Right-skewed — the long tail pulls the mean past the median' : 'Left-skewed — the long tail pulls the mean below the median'}
        </span>
      </div>
    </VizBox>
  );
}

/* ── Correlation scatter demo ──────────────────────────────── */
function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

function CorrelationScatter() {
  const [r, setR] = useState(0.6);
  const n = 45;
  const rand = seededRandom(42);
  const points = Array.from({ length: n }, () => {
    const x = rand() * 2 - 1;
    const noise = rand() * 2 - 1;
    const y = r * x + Math.sqrt(Math.max(0, 1 - r * r)) * noise;
    return { x, y };
  });

  return (
    <VizBox>
      <SliderRow label="r" min={-1} max={1} step={0.1} value={r} onChange={setR} fmt={v => v.toFixed(1)} />
      <div style={{ position: 'relative', height: 190, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden' }}>
        {points.map((p, i) => (
          <div key={i} style={{
            position: 'absolute',
            left: `${((p.x + 1.3) / 2.6) * 100}%`,
            top: `${100 - ((p.y + 1.3) / 2.6) * 100}%`,
            width: 6, height: 6, borderRadius: '50%',
            background: 'var(--color-border-info)', opacity: 0.75,
            transform: 'translate(-50%, -50%)',
          }} />
        ))}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        At r = {r.toFixed(1)}, points {Math.abs(r) > 0.7 ? 'cluster tightly around a line' : Math.abs(r) > 0.3 ? 'show a loose trend' : 'look almost like a random cloud'}.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'sanity', label: 'Data Sanity Checks', sub: 'Dtypes, duplicates & cardinality' },
  { id: 'univariate', label: 'Univariate Analysis', sub: 'Summary stats & distributions' },
  { id: 'multivariate', label: 'Bivariate & Multivariate', sub: 'Relationships between variables' },
  { id: 'profiling', label: 'Automated Profiling', sub: 'One-shot dataset reports' },
];

function SectionSanity() {
  return (
    <div>
      <P>
        Before any chart gets drawn, check the basic health of the dataframe. This is the fastest way
        to catch problems that would otherwise quietly bias every statistic computed afterward.
      </P>

      <H2 c="The First Look" />
      <Table
        heads={['Check', 'What it tells you', 'Pandas call']}
        rows={[
          ['Shape', 'Row and column count — sanity-checks that a load or a join didn’t drop rows', 'df.shape'],
          ['Dtypes', 'Wrong types hide bugs, e.g. a numeric column read in as text', 'df.dtypes / df.info()'],
          ['Duplicates', 'Repeated rows silently double-count in statistics and can leak across a train/test split', 'df.duplicated().sum()'],
          ['Cardinality', 'Near-unique columns (IDs) rarely predict anything; near-constant columns carry no signal', 'df.nunique()'],
        ]}
      />

      <H2 c="Missingness" />
      <P>
        Counting nulls only says how much is missing. Visualizing where they fall says whether the
        gaps are random or structural — an entire block of rows missing the same three columns
        usually means something specific happened (a form field added later, a sensor offline), not
        chance.
      </P>
      <Note color="info" icon="ti-info-circle">
        The <code>missingno</code> library's <Mx>matrix()</Mx>, <Mx>bar()</Mx>, and <Mx>heatmap()</Mx>{' '}
        plots are the fastest way to see missingness patterns across an entire dataframe at once.
      </Note>

      <Code>{`import pandas as pd

df = pd.read_csv("data.csv")
print(df.shape)
print(df.dtypes)
print(df.duplicated().sum())
print(df.nunique())

import missingno as msno
msno.matrix(df)`}</Code>

      <H2 c="Spot the Issues" />
      <P>Same six rows, three different lenses — toggle between them.</P>
      <SanityCheckDemo />

      <Note color="success" icon="ti-arrow-right">
        Found gaps or outliers here? The fix — deletion, imputation, or a treatment strategy —
        lives in Missing Data and Outliers in the next module, Data Cleaning & Feature Engineering.
      </Note>

      <H2 c="Interview Q&A" />
      <QA
        q="Why check for duplicate rows before computing any summary statistics?"
        a="A duplicated row is double-counted in every mean, count, and correlation computed afterward, and if a duplicate ends up on both sides of a train/test split it leaks information the model shouldn't have had — both silently inflate reported performance."
      />
      <QA
        q="A user ID column is unique on every row — isn't that just what an identifier is supposed to do? Why flag it?"
        a="It's expected, but a raw ID rarely carries predictive signal and, if left in as a feature, tree-based models can latch onto it and effectively memorize individual rows rather than learning generalizable patterns — it's usually dropped before modeling, not encoded."
      />
      <QA
        q="What does it usually mean if a missingness matrix shows whole rows missing several columns together, instead of nulls scattered independently?"
        a="It's a hint the data isn't Missing Completely At Random — something structural is happening (a form section that was optional, a device that went offline for a stretch), which changes which imputation strategy is safe to use."
      />
    </div>
  );
}

function SectionUnivariate() {
  return (
    <div>
      <P>
        Once the dataframe checks out, understand each column on its own before looking at how
        columns relate to each other.
      </P>

      <H2 c="Summary Statistics" />
      <Table
        heads={['Column type', 'Call', 'What you get']}
        rows={[
          ['Numeric', 'df.describe()', 'count, mean, std, min, 25/50/75th percentiles, max'],
          ['Categorical', "df['col'].value_counts()", 'frequency of every distinct category, sorted descending'],
        ]}
      />

      <H2 c="Visualizing Distributions" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="Histogram">
          Bins values into ranges and counts them — the fastest way to see a distribution's overall
          shape.
        </Card>
        <Card color="success" title="KDE Plot">
          A smoothed version of the histogram that doesn't depend on a bin-width choice — good for
          comparing shapes across groups.
        </Card>
        <Card color="purple" title="Box Plot">
          Shows the five-number summary (min, quartiles, max) at a glance, and flags points beyond
          1.5 x IQR as candidate outliers.
        </Card>
      </Grid>

      <Code>{`df.describe()
df["category"].value_counts()

df["price"].hist(bins=30)
df["price"].plot(kind="kde")
df.boxplot(column="price")`}</Code>

      <H2 c="Skew, Visualized" />
      <P>Drag the slider — watch the mean chase the tail while the median holds steadier.</P>
      <SkewHistogram />

      <H2 c="Interview Q&A" />
      <QA
        q="Why does the mean diverge from the median as a distribution gets more skewed?"
        a="The mean is pulled by every value including extreme ones in the long tail, while the median only cares about the middle-ranked value — so a handful of very large (or very small) values drags the mean away from where most of the data actually sits."
      />
      <QA
        q="You see a right-skewed feature like income or house price. What would you do before feeding it to a linear model?"
        a="Apply a log (or Box-Cox/Yeo-Johnson) transform to pull the long tail in — linear models and anything assuming roughly-normal residuals perform better on the transformed, more symmetric version."
      />
    </div>
  );
}

function SectionMultivariate() {
  return (
    <div>
      <P>With each column understood individually, the next question is how columns move together.</P>

      <H2 c="Two Variables at a Time" />
      <Table
        heads={['Variable pair', 'Typical plot', 'What it reveals']}
        rows={[
          ['Numeric × Numeric', 'Scatter plot', 'Linear vs. non-linear relationship, clusters, outlier pairs'],
          ['Categorical × Numeric', 'Box plot per category', 'Whether the numeric distribution shifts across groups'],
          ['Categorical × Categorical', 'Cross-tab / stacked bar', 'Whether categories co-occur more or less than chance'],
        ]}
      />

      <H2 c="Many Variables at Once" />
      <Grid cols={2} gap={10}>
        <Card color="info" title="Pair Plot">
          A grid of scatter plots for every pair of numeric columns at once — shows actual shape, not
          just a single summary number.
        </Card>
        <Card color="success" title="Correlation Heatmap">
          Color-codes the pairwise correlation matrix — fast for spotting strongly (co-)related
          columns across a wide dataframe.
        </Card>
      </Grid>

      <Code>{`import seaborn as sns

sns.scatterplot(data=df, x="sqft", y="price")
sns.pairplot(df[["sqft", "price", "age"]])
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")`}</Code>

      <H2 c="Correlation Strength, Visualized" />
      <P>Same 45 points regenerated at a target r — drag to see what a given correlation actually looks like.</P>
      <CorrelationScatter />

      <Note color="warning" icon="ti-alert-triangle">
        Correlation heatmaps only capture <em>linear</em> relationships. Two columns can have a strong
        curved (e.g. quadratic) relationship and still show a correlation near zero — always pair a
        heatmap with actual scatter plots before ruling a relationship out.
      </Note>

      <H2 c="Interview Q&A" />
      <QA
        q="Why prefer a pair plot over a single correlation heatmap?"
        a="A heatmap reduces every relationship to one number, which hides non-linear patterns, clusters, and outlier pairs — a pair plot shows the actual shape of each relationship, at the cost of more screen space."
      />
      <QA
        q="Two features show r = 0.02 on the heatmap. Is it safe to conclude they're unrelated?"
        a="No — Pearson correlation only measures linear association. A quadratic or U-shaped relationship can produce a near-zero linear correlation while still being a strong, useful relationship a scatter plot would reveal immediately."
      />
    </div>
  );
}

function SectionProfiling() {
  return (
    <div>
      <P>
        Manual EDA doesn't scale well to dozens of columns. Automated profiling tools generate a full
        report — stats, distributions, correlations, missingness, and warnings — in one call, as a
        starting checklist rather than a replacement for judgment.
      </P>

      <H2 c="One-Shot Reports" />
      <Table
        heads={['Tool', 'Generates', 'Best for']}
        rows={[
          ['YData Profiling (formerly Pandas Profiling)', 'A full HTML report: per-column stats, correlations, missingness, interaction plots, and flagged warnings', 'A first look at a brand-new dataset'],
          ['Sweetviz', 'A similar HTML report, plus a side-by-side comparison view between two datasets', 'Comparing train vs. test (or before vs. after) for distribution drift'],
        ]}
      />

      <Code>{`from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Dataset Report")
profile.to_file("report.html")

import sweetviz as sv
report = sv.compare([train_df, "Train"], [test_df, "Test"])
report.show_html()`}</Code>

      <Note color="warning" icon="ti-alert-triangle">
        These tools compute pairwise statistics for every column combination, which gets slow fast on
        very wide dataframes (100+ columns) — profile a row-sampled subset first if it's taking too
        long.
      </Note>

      <H2 c="Interview Q&A" />
      <QA
        q="If a tool can auto-generate the whole EDA report, why learn to do it manually at all?"
        a="A profiling report is a checklist of what to look at — which columns are skewed, which pairs are correlated, where the missingness is — but deciding which of those findings actually matter for the problem at hand still needs the manual univariate/multivariate judgment covered earlier in this module."
      />
      <QA
        q="You just generated a profiling report on a new dataset. What's the first section worth reading?"
        a="The warnings section — it's already flagged the columns most likely to need attention (high correlation pairs, high cardinality, heavy skew, unexpected zeros), so it's the fastest path to knowing where to look next."
      />
    </div>
  );
}

const SECTION_MAP = {
  sanity: <SectionSanity />,
  univariate: <SectionUnivariate />,
  multivariate: <SectionMultivariate />,
  profiling: <SectionProfiling />,
};

export default function EDA() {
  const [active, setActive] = useState('sanity');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 02</div>
        <h1 className="page-header-title">Exploratory Data Analysis (EDA)</h1>
        <p className="page-header-subtitle">
          Getting to know a dataset before modeling it — distributions, correlations, and visual
          patterns that shape every decision made in the modules that follow.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={2} />
    </div>
  );
}
