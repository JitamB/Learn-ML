import { useState, useMemo, Fragment } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox } from '../components/ui/Primitives.jsx';
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
function toggleBtnStyle(activeSelf) {
  return {
    padding: '5px 12px', fontSize: 12, borderRadius: 'var(--border-radius-md)',
    border: '1px solid ' + (activeSelf ? 'var(--color-border-info)' : 'var(--color-border-tertiary)'),
    background: activeSelf ? 'var(--color-background-info)' : 'transparent',
    color: activeSelf ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
    cursor: 'pointer',
  };
}

/* ── Matrix Factorization demo: real SGD on a sparse ratings grid ── */
const MF_USERS = 6, MF_ITEMS = 6, MF_K = 2;
const USER_LABELS = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6'];
const ITEM_LABELS = ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E', 'Movie F'];
const KNOWN_MASK = [
  [1, 0, 1, 0, 1, 0],
  [0, 1, 0, 1, 0, 1],
  [1, 0, 0, 1, 0, 1],
  [0, 1, 1, 0, 1, 0],
  [1, 0, 1, 0, 0, 1],
  [0, 1, 0, 1, 1, 0],
];
const TRUE_RATINGS = (() => {
  const rand = seededRandom(65);
  const userBias = Array.from({ length: MF_USERS }, () => randNormalish(rand, 0, 0.55));
  const itemBias = Array.from({ length: MF_ITEMS }, () => randNormalish(rand, 0, 0.55));
  const userLatent = Array.from({ length: MF_USERS }, () => [randNormalish(rand, 0, 1), randNormalish(rand, 0, 1)]);
  const itemLatent = Array.from({ length: MF_ITEMS }, () => [randNormalish(rand, 0, 1), randNormalish(rand, 0, 1)]);
  const grid = [];
  for (let u = 0; u < MF_USERS; u++) {
    const row = [];
    for (let i = 0; i < MF_ITEMS; i++) {
      const interaction = (userLatent[u][0] * itemLatent[i][0] + userLatent[u][1] * itemLatent[i][1]) * 1.8 / 2;
      const noise = randNormalish(rand, 0, 0.35);
      row.push(Math.max(1, Math.min(5, Math.round(3 + userBias[u] + itemBias[i] + interaction + noise))));
    }
    grid.push(row);
  }
  return grid;
})();
const INITIAL_RATINGS = TRUE_RATINGS.map((row, u) => row.map((v, i) => (KNOWN_MASK[u][i] ? v : null)));

function trainMF(ratings) {
  const rand = seededRandom(7);
  const lr = 0.02, lambda = 0.2, epochs = 300;
  const P = Array.from({ length: MF_USERS }, () => Array.from({ length: MF_K }, () => randNormalish(rand, 0, 0.5)));
  const Q = Array.from({ length: MF_ITEMS }, () => Array.from({ length: MF_K }, () => randNormalish(rand, 0, 0.5)));
  const entries = [];
  for (let u = 0; u < MF_USERS; u++) for (let i = 0; i < MF_ITEMS; i++) if (ratings[u][i] != null) entries.push({ u, i, r: ratings[u][i] });
  for (let e = 0; e < epochs; e++) {
    for (const { u, i, r } of entries) {
      let pred = 0; for (let f = 0; f < MF_K; f++) pred += P[u][f] * Q[i][f];
      const err = r - pred;
      for (let f = 0; f < MF_K; f++) {
        const pu = P[u][f], qi = Q[i][f];
        P[u][f] += lr * (err * qi - lambda * pu);
        Q[i][f] += lr * (err * pu - lambda * qi);
      }
    }
  }
  return Array.from({ length: MF_USERS }, (_, u) => Array.from({ length: MF_ITEMS }, (_, i) => {
    let s = 0; for (let f = 0; f < MF_K; f++) s += P[u][f] * Q[i][f]; return s;
  }));
}

function MatrixFactorizationDemo() {
  const [ratings, setRatings] = useState(INITIAL_RATINGS);
  const predicted = useMemo(() => trainMF(ratings), [ratings]);
  const maxAbsDev = Math.max(1, ...predicted.flat().map(v => Math.abs(v - 3)));

  function cycleCell(u, i) {
    if (KNOWN_MASK[u][i]) return;
    setRatings(prev => {
      const cur = prev[u][i];
      const next = cur === null ? 1 : (cur >= 5 ? null : cur + 1);
      const copy = prev.map(row => row.slice());
      copy[u][i] = next;
      return copy;
    });
  }

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
        <button onClick={() => setRatings(INITIAL_RATINGS)} style={toggleBtnStyle(false)}>Reset to sample ratings</button>
        <span style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)' }}>Click any faded cell to cycle a rating 1→5 (click past 5 to clear it again).</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: `70px repeat(${MF_ITEMS}, 1fr)`, gap: 4 }}>
        <div />
        {ITEM_LABELS.map(label => (
          <div key={label} style={{ fontSize: 10, textAlign: 'center', color: 'var(--color-text-tertiary)' }}>{label}</div>
        ))}
        {USER_LABELS.map((uLabel, u) => (
          <Fragment key={uLabel}>
            <div style={{ fontSize: 11, display: 'flex', alignItems: 'center', color: 'var(--color-text-secondary)' }}>{uLabel}</div>
            {ITEM_LABELS.map((_, i) => {
              const known = !!KNOWN_MASK[u][i];
              const userSet = !known && ratings[u][i] != null;
              const intensity = Math.min(1, Math.abs(predicted[u][i] - 3) / maxAbsDev);
              return (
                <div
                  key={i}
                  onClick={() => cycleCell(u, i)}
                  style={{
                    height: 34, borderRadius: 'var(--border-radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 13, fontWeight: known || userSet ? 700 : 400, cursor: known ? 'default' : 'pointer',
                    border: userSet ? '2px solid var(--color-border-info)' : '1px solid var(--color-border-tertiary)',
                    background: known ? 'var(--color-background-secondary)' : (userSet ? 'var(--color-background-primary)' : 'var(--color-background-info)'),
                    opacity: (!known && !userSet) ? (0.35 + intensity * 0.65) : 1,
                    color: known ? 'var(--color-text-primary)' : (userSet ? 'var(--color-text-info)' : 'var(--color-text-tertiary)'),
                  }}
                >
                  {known || userSet ? ratings[u][i] : predicted[u][i].toFixed(1)}
                </div>
              );
            })}
          </Fragment>
        ))}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        Bold cells on a plain background are known ratings; bold cells with a blue border are ratings
        <em> you</em> added; faded cells are the model's live predictions, refit from scratch on
        every change. Add a rating anywhere and watch several <em>other</em> faded cells shift too —
        that propagation through shared latent factors is the entire point of matrix factorization.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'filtering', label: 'Content-Based & Collaborative Filtering', sub: 'Two ways to guess what you\'ll like' },
  { id: 'matrixfact', label: 'Matrix Factorization', sub: 'Latent factors from a sparse matrix' },
  { id: 'choosing', label: 'Choosing a Technique', sub: 'A category-wide comparison' },
];

function SectionFiltering() {
  return (
    <div>
      <P>
        Every recommender system is ultimately trying to predict a rating or a preference — "how
        much would this specific user like this specific item?" — for items that user hasn't
        interacted with yet. The two classical families answer that question very differently.
      </P>

      <H2 c="Content-Based Filtering" />
      <P>
        Recommends items similar to ones a given user already liked, based purely on the{' '}
        <em>items' own attributes</em> — genre, cast, ingredients, tags, whatever describes the
        item itself. It never looks at what anyone else did.
      </P>
      <Table
        heads={['Movie', 'Genre tags', 'This user\'s rating']}
        rows={[
          ['Movie A', 'Action, Sci-Fi', '5'],
          ['Movie B', 'Romance, Drama', '2'],
          ['Movie C', 'Action, Sci-Fi', '?  ← recommend: shares tags with the 5-star Movie A'],
          ['Movie D', 'Romance, Comedy', '?  ← skip: shares tags with the 2-star Movie B instead'],
        ]}
      />

      <H2 c="Collaborative Filtering" />
      <P>
        Ignores item attributes entirely and instead relies purely on <em>patterns of behavior</em>{' '}
        across many users.
      </P>
      <Grid cols={2} gap={10}>
        <Card color="info" title="User-User">Find users with a similar rating history to you, then recommend what they liked that you haven't seen.</Card>
        <Card color="success" title="Item-Item">Find items that tend to get rated similarly by the same people, then recommend items similar to ones you already rated highly.</Card>
      </Grid>
      <Table
        heads={['', 'Movie A', 'Movie B', 'Movie C']}
        rows={[
          ['You', '5', '2', '?'],
          ['User X (similar taste)', '5', '1', '5  ← you\'ve both rated A/B alike, so X\'s love of C is a strong signal'],
          ['User Y (different taste)', '1', '5', '2  ← ignored: Y\'s taste doesn\'t track yours'],
        ]}
      />

      <H2 c="The Cold-Start Problem" />
      <Note color="warning" icon="ti-alert-triangle">
        A brand-new item has no ratings yet, so collaborative filtering has nothing to work with —
        but content-based filtering can still recommend it immediately from its attributes alone. A
        brand-new <em>user</em> is worse: with zero rating history, neither approach has much to go
        on, which is why real production systems often blend both (and fall back to simple
        popularity rankings) specifically to cover this gap.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why can content-based filtering recommend a brand-new item immediately, while collaborative filtering can't?"
          a="Content-based filtering only needs the new item's own attributes (genre, tags, description) to compare it against a user's known preferences — those attributes exist the moment the item is added. Collaborative filtering has no signal to work with until at least some users have actually interacted with that item, since its entire basis is patterns of behavior across users, not item descriptions." />
      <QA q="What's a practical downside of pure content-based filtering, even once the cold-start problem is solved?"
          a="It tends to over-specialize — a user who only ever gets recommended near-identical items to what they've already rated highly rarely gets introduced to something genuinely new but still enjoyable, a phenomenon often called a 'filter bubble.' Collaborative filtering, by drawing on other users' behavior, is more likely to surface unexpected but relevant items outside a user's established pattern." />
      <QA q="Why might item-item collaborative filtering be preferred over user-user in a system with millions of users but far fewer items?"
          a="Item-item similarity only needs to be computed and stored across the (much smaller) set of items, and item-to-item relationships tend to be far more stable over time than any individual user's evolving taste — recomputing user-user similarity across millions of users is both more expensive and needs to be refreshed more often as behavior drifts." />
    </div>
  );
}

function SectionMatrixFactorization() {
  return (
    <div>
      <P>
        Matrix factorization reframes "recommend items" as filling in the blanks of a giant,
        mostly-empty spreadsheet: rows are users, columns are items, and each cell is a rating —
        known for the handful of interactions that actually happened, blank everywhere else.
      </P>

      <H2 c="Latent Factors" />
      <P>
        The key idea: assume every user and every item can be described by a short vector of hidden
        ("latent") numbers — not genres or demographics with obvious meaning, just directions the
        algorithm discovers on its own, that might loosely correspond to things like "how much
        action vs. romance" or "how mainstream vs. niche." A predicted rating is just the dot
        product of a user's vector and an item's vector.
      </P>
      <Mx block>{`  rating(user u, item i)  ≈  Pᵤ · Qᵢ  =  Σ (Pᵤ,f · Qᵢ,f)  over f latent factors

  Learn P and Q by minimizing squared error on the KNOWN ratings only,
  with a regularization penalty to stop the vectors from growing
  arbitrarily large just to memorize the training ratings:

  loss = Σ (known)  (rᵤᵢ − Pᵤ·Qᵢ)²  +  λ(‖Pᵤ‖² + ‖Qᵢ‖²)`}</Mx>
      <P>
        This is exactly the SVD (Singular Value Decomposition) idea from Math Foundations, adapted
        to work when most of the matrix is missing rather than fully known.
      </P>
      <Note color="info" icon="ti-info-circle">
        Because both a user's and an item's rating for <em>any</em> cell route through the same
        small set of shared latent factors, adding just one new known rating nudges those factors —
        which then shifts the model's predictions for every other cell that shares either that user
        or that item, and often cells that share neither, indirectly.
      </Note>

      <H2 c="Try It — Add a Rating, Watch Predictions Update" />
      <P>6 users × 6 movies. A real SGD fit (retrained from scratch on every edit — it's cheap enough) fills in every blank cell live.</P>
      <MatrixFactorizationDemo />
      <Note color="warning" icon="ti-alert-triangle">
        With only a handful of known ratings per user/item, this tiny problem is <em>severely</em>{' '}
        underdetermined — regularization (<Mx>λ</Mx>) has to work harder than usual here to keep
        predictions inside a sane 1-5 range instead of extrapolating to nonsense. Real systems have
        far more ratings per user, which naturally constrains the fit much better.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from surprise import SVD, Dataset, Reader

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["user", "item", "rating"]], reader)

algo = SVD(n_factors=20, reg_all=0.05)
algo.fit(data.build_full_trainset())
algo.predict(uid="user_42", iid="movie_7")`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why does matrix factorization need regularization more urgently than a typical supervised model with a large, dense training set?"
          a="A ratings matrix is usually extremely sparse — most user-item pairs are unobserved — so the number of known entries can be small relative to the number of free parameters in P and Q. Without a strong-enough regularization penalty, the optimizer can drive the latent vectors to whatever values minimize error on the sparse known entries, including values that produce wildly implausible predictions for anything unobserved." />
      <QA q="What do the 'latent factors' in matrix factorization actually represent?"
          a="Nothing guaranteed or human-labeled — they're just whatever numeric directions the optimization discovers that best explain the pattern of known ratings, found automatically rather than engineered. They sometimes turn out to loosely correlate with interpretable concepts (like genre preference) purely as a side effect of what happens to explain the data well, but there's no guarantee of that, and no label attached to any individual factor." />
      <QA q="Why is matrix factorization considered a form of collaborative filtering rather than content-based filtering?"
          a="It never looks at any item's actual attributes (genre, description, etc.) — every latent factor is learned purely from the pattern of who-rated-what, which is the defining trait of collaborative filtering. It shares collaborative filtering's cold-start weakness for exactly this reason: a brand-new item or user with no ratings yet has no observed pattern to learn a latent vector from." />
      <QA q="How does the choice of the number of latent factors (k) trade off against overfitting?"
          a="A very small k forces users and items into an overly simplistic shared representation, likely underfitting real preference structure (high bias). A very large k gives the model enough free parameters to memorize the sparse known ratings almost exactly, at the cost of generalizing poorly to unseen user-item pairs (high variance) — k is typically tuned via validation, same as any other capacity hyperparameter." />
    </div>
  );
}

const CHOOSING_ROWS = [
  ['K-Means', 'Numeric feature vectors', 'Cluster assignment + centroids', 'K (number of clusters)', 'Fast — scales well', 'Customer segmentation, quick exploratory grouping'],
  ['Hierarchical Clustering', 'Numeric vectors (or a distance matrix)', 'Full dendrogram, cut at any level', 'Linkage method + cut height', 'Slow — naively O(n³)', 'Small datasets where nested structure matters'],
  ['DBSCAN', 'Numeric feature vectors', 'Cluster assignment + explicit noise label', 'epsilon, min_samples', 'Moderate', 'Arbitrarily-shaped clusters, data with real outliers'],
  ['PCA', 'Numeric vectors (standardized)', 'Lower-dimensional linear projection', 'Number of components to keep', 'Fast — one-time decomposition', 'Compression, noise reduction, preprocessing'],
  ['t-SNE', 'Numeric feature vectors', '2D/3D layout for visualization only', 'Perplexity', 'Slow — many O(n²) iterations', 'Visually exploring high-dimensional structure'],
  ['Apriori', 'Transactional (basket) data', 'Ranked rules with support/confidence/lift', 'min_support, min_confidence', 'Can be expensive at scale', 'Market basket analysis, cross-sell recommendations'],
  ['Isolation Forest', 'Numeric feature vectors', 'Continuous anomaly score per point', 'Number of trees, contamination', 'Fast — O(n log n) to build', 'Fraud/outlier detection with no labeled anomalies'],
  ['Matrix Factorization', 'Sparse user-item matrix', 'Dense predicted rating for every cell', 'Latent dimension k, regularization', 'Moderate — scales with known entries', 'Recommender systems at scale'],
];

function SectionChoosing() {
  return (
    <div>
      <P>
        A quick reference across every technique covered in this category — unlike supervised
        learning, there's no single "accuracy" number to compare these on, since most have no
        ground truth to check against at all. The right choice depends almost entirely on what kind
        of data you have and what shape of answer you actually need.
      </P>

      <H2 c="Technique Comparison" />
      <Table
        heads={['Technique', 'Input Needed', 'Output', 'Key Hyperparameter(s)', 'Compute Cost', 'Typical Use Case']}
        rows={CHOOSING_ROWS}
      />

      <Note color="success" icon="ti-bulb">
        A reasonable default workflow: if the data is unlabeled feature vectors and the goal is
        grouping, start with K-Means for a fast first pass, then reach for DBSCAN if the clusters
        turn out to be non-spherical or noisy. If the goal is understanding/visualizing structure
        rather than grouping it, reach for PCA first (it's nearly free), and only bring in t-SNE
        once a 2D picture of genuinely non-linear structure is specifically needed.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why is there no single universal metric for comparing unsupervised techniques, the way accuracy or F1 works for supervised ones?"
          a="Supervised metrics compare a prediction against a known correct answer, which unsupervised learning by definition doesn't have — there's no ground-truth cluster label, no 'correct' 2D embedding, no labeled anomaly to check against. Evaluation instead relies on internal consistency measures (like silhouette score for clustering) or downstream task performance, which don't generalize across every technique in this category the way a single classification metric does." />
      <QA q="A dataset is unlabeled numeric sensor data with a handful of clearly broken/miscalibrated readings. Which technique here would you reach for first, and why?"
          a="Isolation Forest — it's built specifically to flag points that don't fit the learned pattern of 'normal' readings, requires no labeled examples of what a broken reading looks like, and is fast enough to run on sensor-scale data without much tuning. Clustering algorithms could be adapted to flag outliers indirectly, but that's not what they're optimized for." />
      <QA q="Why would you reach for PCA before t-SNE even though t-SNE often produces more visually striking cluster separation?"
          a="PCA is a cheap, deterministic, linear operation with a clear mathematical meaning (variance explained) — it's a reasonable first check on almost any numeric dataset, and is often used as a preprocessing step before t-SNE itself, to cut noise and dimensionality before the much more expensive non-linear optimization. Reaching straight for t-SNE without first understanding the linear structure risks over-interpreting an embedding whose distances and cluster sizes aren't reliably meaningful in the first place." />
    </div>
  );
}

const SECTION_MAP = {
  filtering: <SectionFiltering />,
  matrixfact: <SectionMatrixFactorization />,
  choosing: <SectionChoosing />,
};

export default function RecommenderSystems() {
  const [active, setActive] = useState('filtering');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 12</div>
        <h1 className="page-header-title">Recommender Systems</h1>
        <p className="page-header-subtitle">
          How "you might also like" actually works — including matrix factorization, which reuses
          the SVD from the Math Foundations prerequisites.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={12} />
    </div>
  );
}
