import { useState, useMemo, useRef } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow, Badge } from '../components/ui/Primitives.jsx';
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
const CLUSTER_TEXT = ['var(--color-text-info)', 'var(--color-text-danger)', 'var(--color-text-success)', 'var(--color-text-warning)', 'var(--color-text-purple)'];
function distSq(ax, ay, bx, by) { const dx = ax - bx, dy = ay - by; return dx * dx + dy * dy; }

/* ── Shared dataset: 3 blobs (K-Means uses all 45, Hierarchical subsamples 15) ── */
const KMEANS_POINTS = (() => {
  const randA = seededRandom(11), randB = seededRandom(47);
  const centers = [[2.5, 2.7], [7.6, 3.0], [5.0, 8.0]];
  const pts = [];
  centers.forEach(([cx, cy]) => {
    for (let i = 0; i < 15; i++) {
      pts.push({
        x: Math.max(0.3, Math.min(9.7, randNormalish(randA, cx, 1.6))),
        y: Math.max(0.3, Math.min(9.7, randNormalish(randB, cy, 1.6))),
      });
    }
  });
  return pts;
})();

/* ── K-Means demo: real Lloyd's algorithm, click-to-place centroids ── */
function KMeansDemo() {
  const [centroids, setCentroids] = useState([]);
  const [assignments, setAssignments] = useState(null);
  const [converged, setConverged] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const boxRef = useRef(null);

  const handleClick = (e) => {
    if (centroids.length >= 4 || converged) return;
    const rect = boxRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * 10;
    const y = 10 - (e.clientY - rect.top) / rect.height * 10;
    setCentroids(prev => [...prev, { x, y }]);
  };

  const step = () => {
    if (centroids.length === 0 || converged) return;
    const newAssign = KMEANS_POINTS.map(p => {
      let best = 0, bestD = Infinity;
      centroids.forEach((c, ci) => { const d = distSq(p.x, p.y, c.x, c.y); if (d < bestD) { bestD = d; best = ci; } });
      return best;
    });
    const newCentroids = centroids.map((c, ci) => {
      const members = KMEANS_POINTS.filter((_, i) => newAssign[i] === ci);
      if (members.length === 0) return c; // freeze an emptied cluster in place — avoids NaN, contributes 0 to WCSS either way
      return {
        x: members.reduce((s, p) => s + p.x, 0) / members.length,
        y: members.reduce((s, p) => s + p.y, 0) / members.length,
      };
    });
    const moved = newCentroids.some((c, i) => Math.abs(c.x - centroids[i].x) > 0.01 || Math.abs(c.y - centroids[i].y) > 0.01);
    if (!moved && assignments !== null) setConverged(true);
    setAssignments(newAssign);
    setCentroids(newCentroids);
    setStepCount(s => s + 1);
  };

  const reset = () => { setCentroids([]); setAssignments(null); setConverged(false); setStepCount(0); };

  const wcss = useMemo(() => {
    if (!assignments) return null;
    return KMEANS_POINTS.reduce((sum, p, i) => sum + distSq(p.x, p.y, centroids[assignments[i]].x, centroids[assignments[i]].y), 0);
  }, [assignments, centroids]);

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
        <button onClick={step} disabled={centroids.length === 0 || converged} style={{ ...toggleBtnStyle(false), opacity: (centroids.length === 0 || converged) ? 0.4 : 1 }}>Step</button>
        <button onClick={reset} style={toggleBtnStyle(false)}>Reset</button>
        <span style={{ fontSize: 12, color: 'var(--color-text-tertiary)' }}>
          {centroids.length === 0 ? 'Click the plot to place up to 4 centroids' :
            converged ? `Converged after ${stepCount} steps` :
            `${centroids.length} centroid${centroids.length === 1 ? '' : 's'} placed — click Step to assign & update`}
        </span>
        {wcss !== null && <Badge color="info">WCSS: {wcss.toFixed(1)}</Badge>}
      </div>
      <div
        ref={boxRef}
        onClick={handleClick}
        style={{
          position: 'relative', height: 260, border: '1px solid var(--color-border-tertiary)',
          borderRadius: 'var(--border-radius-md)', overflow: 'hidden', background: 'var(--color-background-secondary)',
          cursor: (centroids.length < 4 && !converged) ? 'crosshair' : 'default',
        }}
      >
        {KMEANS_POINTS.map((p, i) => {
          const ci = assignments ? assignments[i] : null;
          return (
            <div key={i} style={{
              position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
              transform: 'translate(-50%,-50%)', width: 7, height: 7, borderRadius: '50%',
              background: ci === null ? 'var(--color-text-tertiary)' : CLUSTER_TEXT[ci],
            }} />
          );
        })}
        {centroids.map((c, i) => (
          <div key={i} style={{
            position: 'absolute', left: `${c.x / 10 * 100}%`, top: `${(1 - c.y / 10) * 100}%`,
            transform: 'translate(-50%,-50%)', width: 18, height: 18, borderRadius: '50%',
            border: `3px solid ${CLUSTER_TEXT[i]}`, background: 'var(--color-background-primary)',
            boxShadow: 'var(--shadow-sm)',
          }} />
        ))}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        {converged
          ? "WCSS never increased on any step — that's not a coincidence, it's a mathematical guarantee of Lloyd's algorithm. Try placing centroids differently and see if you land on a different final WCSS."
          : 'Rings = centroids, dots = data points colored by current assignment (gray = unassigned). Each Step alternates: assign every point to its nearest centroid, then move each centroid to the mean of its assigned points.'}
      </div>
    </VizBox>
  );
}

/* ── Hierarchical Clustering demo: real agglomerative fit, 4 linkages, dendrogram ── */
const HIER_POINTS = [0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 30, 31, 32, 33, 34].map(i => KMEANS_POINTS[i]);
function euclid(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }
function centroidOf(members) {
  return { x: members.reduce((s, p) => s + p.x, 0) / members.length, y: members.reduce((s, p) => s + p.y, 0) / members.length };
}
function clusterDistance(linkage, A, B) {
  if (linkage === 'single') { let m = Infinity; for (const a of A) for (const b of B) m = Math.min(m, euclid(a, b)); return m; }
  if (linkage === 'complete') { let m = 0; for (const a of A) for (const b of B) m = Math.max(m, euclid(a, b)); return m; }
  if (linkage === 'average') { let sum = 0, n = 0; for (const a of A) for (const b of B) { sum += euclid(a, b); n++; } return sum / n; }
  const cA = centroidOf(A), cB = centroidOf(B); // ward
  return (A.length * B.length) / (A.length + B.length) * ((cA.x - cB.x) ** 2 + (cA.y - cB.y) ** 2);
}
function buildMergeSequence(points, linkage) {
  let clusters = points.map((p, i) => ({ id: i, members: [p], pointIndices: [i] }));
  const merges = [];
  let nextId = points.length;
  while (clusters.length > 1) {
    let best = null;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const d = clusterDistance(linkage, clusters[i].members, clusters[j].members);
        if (!best || d < best.d) best = { i, j, d };
      }
    }
    const A = clusters[best.i], B = clusters[best.j];
    const merged = { id: nextId++, members: [...A.members, ...B.members], pointIndices: [...A.pointIndices, ...B.pointIndices] };
    merges.push({ leftId: A.id, rightId: B.id, dist: best.d, newId: merged.id, leftPointIndices: A.pointIndices, rightPointIndices: B.pointIndices });
    clusters = clusters.filter((_, idx) => idx !== best.i && idx !== best.j);
    clusters.push(merged);
  }
  return merges;
}
function buildDendrogramTree(points, merges) {
  const nodeById = {};
  points.forEach((p, i) => { nodeById[i] = { id: i, isLeaf: true, height: 0 }; });
  merges.forEach(m => { nodeById[m.newId] = { id: m.newId, isLeaf: false, left: nodeById[m.leftId], right: nodeById[m.rightId], height: m.dist }; });
  return nodeById[merges[merges.length - 1].newId];
}
function layoutDendrogram(root, maxHeight) {
  let nextSlot = 0;
  (function assignLeafSlots(node) {
    if (node.isLeaf) { node.xSlot = nextSlot++; return; }
    assignLeafSlots(node.left); assignLeafSlots(node.right);
  })(root);
  (function assignInternalSlots(node) {
    if (node.isLeaf) return node.xSlot;
    node.xSlot = (assignInternalSlots(node.left) + assignInternalSlots(node.right)) / 2;
    return node.xSlot;
  })(root);
  const numLeaves = Math.max(1, nextSlot);
  const nodes = [], edges = [];
  (function collect(node) {
    node.xPct = (node.xSlot + 0.5) / numLeaves * 100;
    node.yPct = maxHeight > 0 ? (1 - node.height / maxHeight) * 100 : 100;
    nodes.push(node);
    if (!node.isLeaf) { edges.push([node, node.left]); edges.push([node, node.right]); collect(node.left); collect(node.right); }
  })(root);
  return { nodes, edges };
}
function unionFind(n) {
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
  function union(a, b) { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; }
  return { find, union };
}
function flatClusters(points, merges, threshold) {
  const { find, union } = unionFind(points.length);
  merges.forEach(m => { if (m.dist <= threshold) union(m.leftPointIndices[0], m.rightPointIndices[0]); });
  const rootOf = points.map((_, i) => find(i));
  const uniqueRoots = [...new Set(rootOf)].sort((a, b) => a - b);
  const rootToIdx = {}; uniqueRoots.forEach((r, i) => { rootToIdx[r] = i; });
  return { clusterOf: rootOf.map(r => rootToIdx[r]), count: uniqueRoots.length };
}
const LINKAGES = [{ id: 'single', label: 'Single' }, { id: 'complete', label: 'Complete' }, { id: 'average', label: 'Average' }, { id: 'ward', label: 'Ward' }];

function HierarchicalDemo() {
  const [linkage, setLinkage] = useState('ward');
  const [thresholdP, setThresholdP] = useState(0.3);
  const merges = useMemo(() => buildMergeSequence(HIER_POINTS, linkage), [linkage]);
  const maxDist = Math.max(...merges.map(m => m.dist));
  const minDist = Math.min(...merges.map(m => m.dist));
  const floor = minDist > 1e-6 ? minDist : 0.01 * maxDist;
  const threshold = thresholdP <= 0 ? 0 : floor * Math.pow(maxDist / floor, thresholdP);
  const { clusterOf, count } = useMemo(() => flatClusters(HIER_POINTS, merges, threshold), [merges, threshold]);
  const tree = useMemo(() => buildDendrogramTree(HIER_POINTS, merges), [merges]);
  const { nodes, edges } = useMemo(() => layoutDendrogram(tree, maxDist), [tree, maxDist]);
  const cutYPct = maxDist > 0 ? (1 - threshold / maxDist) * 100 : 100;
  const colorable = count <= 5;

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        {LINKAGES.map(l => (
          <button key={l.id} onClick={() => setLinkage(l.id)} style={toggleBtnStyle(linkage === l.id)}>{l.label}</button>
        ))}
      </div>
      <SliderRow label="Distance threshold" min={0} max={1} step={0.005} value={thresholdP} onChange={setThresholdP} fmt={() => threshold.toFixed(2)} />
      <Grid cols={2} gap={12}>
        <div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Dendrogram ({count} cluster{count === 1 ? '' : 's'} at this cut)</div>
          <div style={{ position: 'relative', height: 220, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)' }}>
            <svg viewBox="0 0 100 220" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
              {edges.map(([parent, child], i) => (
                <polyline key={i}
                  points={`${child.xPct},${child.yPct * 2.2} ${child.xPct},${parent.yPct * 2.2} ${parent.xPct},${parent.yPct * 2.2}`}
                  fill="none" stroke="var(--color-border-secondary)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
              ))}
              <line x1={0} y1={cutYPct * 2.2} x2={100} y2={cutYPct * 2.2} stroke="var(--color-text-danger)" strokeWidth={1} strokeDasharray="4,3" vectorEffect="non-scaling-stroke" />
            </svg>
            {nodes.filter(n => n.isLeaf).map((node, i) => (
              <div key={i} style={{
                position: 'absolute', left: `${node.xPct}%`, top: node.yPct * 2.2,
                transform: 'translate(-50%,-50%)', width: 7, height: 7, borderRadius: '50%',
                background: colorable ? CLUSTER_TEXT[clusterOf[node.id] % CLUSTER_TEXT.length] : 'var(--color-text-tertiary)',
              }} />
            ))}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Scatter, colored by current cut</div>
          <div style={{ position: 'relative', height: 220, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden', background: 'var(--color-background-secondary)' }}>
            {HIER_POINTS.map((p, i) => (
              <div key={i} style={{
                position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
                transform: 'translate(-50%,-50%)', width: 9, height: 9, borderRadius: '50%',
                background: colorable ? CLUSTER_TEXT[clusterOf[i] % CLUSTER_TEXT.length] : 'var(--color-text-tertiary)',
              }} />
            ))}
          </div>
        </div>
      </Grid>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 10 }}>
        {linkage === 'ward'
          ? "Ward's merge cost scales with cluster size, so late merges of well-separated clusters cost dramatically more than early ones — notice how wide a stretch of this slider keeps 3 clusters stable once they're each internally tight. That's real Ward behavior, not a stuck slider."
          : linkage === 'single'
          ? 'Single linkage merges whichever two points are closest, even across otherwise-distant clusters — watch it "chain," absorbing one point at a time rather than merging whole groups at once.'
          : 'Drag the threshold to cut the tree at different heights — the dashed red line shows exactly where, and the scatter recolors to match.'}
      </div>
    </VizBox>
  );
}

/* ── DBSCAN demo: real core/border/noise classification on two interleaving moons ── */
const DBSCAN_POINTS = (() => {
  // extraGap widens the base two-moons formula's natural minimum gap (0.5*radius) so the sweet
  // spot isn't razor-thin once jitter is added — verified numerically before shipping.
  const S = 2.5, shiftX = 3.75, shiftY = 4.375, jitterStd = 0.20, extraGap = 1.0;
  const rand1 = seededRandom(401), rand2 = seededRandom(402);
  const pts = [];
  for (let i = 0; i < 24; i++) {
    const t = (i / 23) * Math.PI;
    pts.push({ x: shiftX + S * Math.cos(t) + randNormalish(rand1, 0, jitterStd), y: shiftY + S * Math.sin(t) + randNormalish(rand2, 0, jitterStd) });
  }
  const rand3 = seededRandom(403), rand4 = seededRandom(404);
  for (let i = 0; i < 24; i++) {
    const t = (i / 23) * Math.PI;
    pts.push({ x: shiftX + S - S * Math.cos(t) + randNormalish(rand3, 0, jitterStd), y: shiftY - S * Math.sin(t) + S * 0.5 - extraGap + randNormalish(rand4, 0, jitterStd) });
  }
  const crescentPts = pts.slice();
  const noiseSeeds = [101, 202, 303, 404, 505, 606, 707, 808];
  const noise = [];
  noiseSeeds.forEach(seed => {
    const r = seededRandom(seed);
    let pt, tries = 0;
    do {
      pt = { x: 0.5 + r() * 9.0, y: 0.5 + r() * 7.5 };
      tries++;
    } while (tries < 300 && (crescentPts.some(c => Math.hypot(c.x - pt.x, c.y - pt.y) < 1.5) || noise.some(n => Math.hypot(n.x - pt.x, n.y - pt.y) < 1.6)));
    noise.push(pt);
  });
  return pts.concat(noise);
})();
function dbscanFit(points, eps, minSamples) {
  const n = points.length;
  const neighborLists = points.map(p => {
    const list = [];
    for (let j = 0; j < n; j++) if (Math.hypot(p.x - points[j].x, p.y - points[j].y) <= eps) list.push(j);
    return list;
  });
  const isCore = neighborLists.map(list => list.length >= minSamples);
  const clusterId = new Array(n).fill(-1);
  let current = 0;
  for (let i = 0; i < n; i++) {
    if (clusterId[i] !== -1 || !isCore[i]) continue;
    clusterId[i] = current;
    const queue = [...neighborLists[i]];
    while (queue.length) {
      const j = queue.shift();
      if (clusterId[j] === -1) { clusterId[j] = current; if (isCore[j]) queue.push(...neighborLists[j]); }
    }
    current++;
  }
  const kind = points.map((_, i) => clusterId[i] === -1 ? 'noise' : (isCore[i] ? 'core' : 'border'));
  return { clusterId, kind, numClusters: current };
}

function DBSCANDemo() {
  const [eps, setEps] = useState(1.2);
  const [minSamples, setMinSamples] = useState(4);
  const { clusterId, kind, numClusters } = useMemo(() => dbscanFit(DBSCAN_POINTS, eps, minSamples), [eps, minSamples]);
  const coreCount = kind.filter(k => k === 'core').length;
  const borderCount = kind.filter(k => k === 'border').length;
  const noiseCount = kind.filter(k => k === 'noise').length;

  return (
    <VizBox>
      <SliderRow label="epsilon (ε)" min={0.3} max={2.5} step={0.02} value={eps} onChange={setEps} fmt={v => v.toFixed(2)} />
      <SliderRow label="min_samples" min={2} max={10} step={1} value={minSamples} onChange={setMinSamples} fmt={v => `${v}`} />
      <div style={{ position: 'relative', height: 260, marginTop: 14, border: '1px solid var(--color-border-tertiary)', borderRadius: 'var(--border-radius-md)', overflow: 'hidden', background: 'var(--color-background-secondary)' }}>
        {DBSCAN_POINTS.map((p, i) => {
          if (kind[i] === 'noise') {
            return (
              <div key={i} style={{
                position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`,
                transform: 'translate(-50%,-50%)', fontSize: 11, fontWeight: 700, color: 'var(--color-text-tertiary)',
              }}>✕</div>
            );
          }
          const color = CLUSTER_TEXT[clusterId[i] % CLUSTER_TEXT.length];
          return kind[i] === 'core' ? (
            <div key={i} style={{ position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`, transform: 'translate(-50%,-50%)', width: 8, height: 8, borderRadius: '50%', background: color }} />
          ) : (
            <div key={i} style={{ position: 'absolute', left: `${p.x / 10 * 100}%`, top: `${(1 - p.y / 10) * 100}%`, transform: 'translate(-50%,-50%)', width: 8, height: 8, borderRadius: '50%', border: `2px solid ${color}`, background: 'var(--color-background-primary)' }} />
          );
        })}
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 8 }}>
        {numClusters} cluster{numClusters === 1 ? '' : 's'} found — {coreCount} core (filled), {borderCount} border (ringed), {noiseCount} noise (✕).{' '}
        {eps < 0.5 ? 'Epsilon is too small — almost everything looks isolated and gets marked noise.'
          : eps > 1.8 ? 'Epsilon is now large enough to bridge the gap between the two crescents into one blob.'
          : 'This is close to the sweet spot — two clean crescents, with the scattered noise points correctly excluded from both.'}
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'kmeans', label: 'K-Means', sub: 'Centroid-based partitioning' },
  { id: 'hierarchical', label: 'Hierarchical Clustering', sub: 'Bottom-up merging' },
  { id: 'dbscan', label: 'DBSCAN', sub: 'Density-based, arbitrary shapes' },
];

function SectionKMeans() {
  return (
    <div>
      <P>
        K-Means asks a deceptively simple question: if you had to summarize a dataset with just{' '}
        <Mx>K</Mx> points, where should those points go so that every data point is as close as
        possible to the point summarizing it? Those <Mx>K</Mx> summary points are the centroids,
        and "close" is measured by ordinary Euclidean distance.
      </P>

      <H2 c="The Algorithm" />
      <Mx block>{`  1. Randomly initialize K centroids
  2. Assign each point to its nearest centroid  ("assignment step")
  3. Move each centroid to the mean of the points now assigned to it  ("update step")
  4. Repeat 2-3 until no assignment changes (convergence)`}</Mx>
      <P>
        A worked-by-hand round makes this concrete. Say two centroids start at{' '}
        <Mx>C₁=(2,2)</Mx> and <Mx>C₂=(8,8)</Mx>, with three points{' '}
        <Mx>A=(1,1)</Mx>, <Mx>B=(4,2)</Mx>, <Mx>C=(9,7)</Mx>:
      </P>
      <Mx block>{`  Assign:  d(A,C₁)=1.41, d(A,C₂)=9.90  → A joins C₁
           d(B,C₁)=2.00, d(B,C₂)=7.21  → B joins C₁
           d(C,C₁)=8.60, d(C,C₂)=1.41  → C joins C₂

  Update:  C₁ ← mean(A,B) = ((1+4)/2, (1+2)/2) = (2.5, 1.5)
           C₂ ← mean(C)   = (9, 7)`}</Mx>
      <P>Both centroids moved toward the points they now represent — that's one full round.</P>

      <H2 c="Why It's Guaranteed to Converge" />
      <P>
        Every assignment step can only decrease (or leave unchanged) the total squared distance
        from points to their centroid, since each point picks its <em>closest</em> option. Every
        update step also only decreases it, since the mean is provably the point that minimizes
        summed squared distance to a fixed set of points. A quantity that can only ever decrease,
        bounded below by zero, must eventually stop changing — that's the whole proof.
      </P>
      <Note color="warning" icon="ti-alert-triangle">
        <strong>Limitations:</strong> K-Means assumes clusters are roughly spherical and similarly
        sized — it has no way to represent an elongated or oddly-shaped group. It's also highly
        sensitive to where centroids start: a bad initialization can converge to a genuinely worse
        (higher-WCSS) local optimum than a better one, which is why real implementations
        (scikit-learn's <code>k-means++</code>) run several random initializations and keep the
        best result rather than trusting a single run.
      </Note>

      <H2 c="Try It — Place Centroids, Then Step" />
      <P>45 points forming 3 loose blobs. Click to drop up to 4 centroids, then click Step repeatedly to watch assignment and update alternate until convergence.</P>
      <KMeansDemo />

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(X)

print("Centroids:", km.cluster_centers_)
print("WCSS (inertia):", km.inertia_)`}</Code>

      <Note color="success" icon="ti-arrow-right">
        Choosing K itself — via the elbow method or silhouette score — along with judging any
        clustering when there's no ground truth to check against, is covered in "Metrics" under
        Model Evaluation & Validation.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why is K-Means guaranteed to converge, and does it always converge to the globally best solution?"
          a="It's guaranteed to converge because within-cluster sum of squares (WCSS) can only decrease or stay the same at both the assignment step (each point picks its closest centroid) and the update step (the mean minimizes summed squared distance to a fixed set of points) — a bounded, monotonically non-increasing quantity must eventually stop changing. It is NOT guaranteed to reach the global optimum, though — it can get stuck in a local minimum depending on initial centroid placement, which is why real implementations run multiple random initializations and keep the best result." />
      <QA q="Why does K-Means struggle with clusters of very different sizes or densities?"
          a="Because it only minimizes squared distance to a centroid, K-Means implicitly assumes clusters are roughly spherical and similarly sized — it will happily 'steal' points from a large, sparse cluster to feed a small, dense one nearby, since that can lower total WCSS even though it's the wrong grouping. Density-based methods like DBSCAN don't share this assumption." />
      <QA q="What does an empty cluster (zero points assigned) during an update step imply, and how should it be handled?"
          a="It means no point in the dataset was closer to that centroid than to any other — often because another centroid started very close by and 'won' every nearby point. A common, safe fix is to simply leave that centroid's position unchanged for the round rather than computing an undefined mean of zero points; it will typically start winning points again once the other centroids move away in later rounds." />
      <QA q="How does the choice of K affect WCSS, and why can't you just pick the K that minimizes it?"
          a="WCSS decreases monotonically as K increases, hitting zero when K equals the number of data points (every point is its own cluster) — so minimizing WCSS alone always favors the largest K, which is a meaningless answer. This is exactly why K is chosen by looking for a bend in the WCSS-vs-K curve (the elbow method) or by a separate metric like silhouette score, not by minimizing WCSS directly." />
    </div>
  );
}

function SectionHierarchical() {
  return (
    <div>
      <P>
        Instead of committing to one specific number of clusters up front, hierarchical clustering
        builds an entire family tree of groupings — from every point as its own cluster, up to one
        cluster containing everything — and lets you pick any cut through that tree afterward.
      </P>

      <H2 c="Agglomerative (Bottom-Up) Clustering" />
      <Mx block>{`  1. Start with every point as its own cluster
  2. Find the two closest clusters (by whichever linkage rule is chosen)
  3. Merge them into one
  4. Repeat 2-3 until only one cluster remains`}</Mx>

      <H2 c="Linkage Criteria" />
      <Table
        heads={['Linkage', 'Distance Between Two Clusters', 'Tendency']}
        rows={[
          ['Single', 'The closest pair of points, one from each cluster', 'Can "chain" — absorbs one nearby point at a time'],
          ['Complete', 'The farthest pair of points, one from each cluster', 'Favors compact, evenly-sized clusters'],
          ['Average', 'The mean distance across every cross-cluster pair', 'A middle ground between single and complete'],
          ['Ward', 'The increase in total within-cluster variance the merge would cause', "Minimizes variance directly — scikit-learn's default"],
        ]}
      />
      <P>
        Ward's merge cost has a clean closed form for two clusters A and B:{' '}
        <Mx>{'cost = (|A|·|B|)/(|A|+|B|) · ‖centroidA − centroidB‖²'}</Mx>. For a 2-point cluster
        centered at (1,1) and a 3-point cluster centered at (4,5):
      </P>
      <Mx block>{`  cost = (2·3)/(2+3) · [(4-1)² + (5-1)²]
       = 1.2 · [9 + 16]
       = 1.2 · 25
       = 30`}</Mx>

      <H2 c="Dendrograms" />
      <P>
        A dendrogram plots every merge as a horizontal join, at a height equal to that merge's
        distance. Dragging a horizontal cut line down through it is exactly equivalent to asking
        "what did the clustering look like when clusters were still at least this different?" —
        the lower the cut, the more (and smaller) clusters survive.
      </P>

      <H2 c="Try It — Compare Linkages" />
      <P>Same 15 points (a small sample from the K-Means blobs) — switch linkage and drag the threshold to cut the tree at different heights.</P>
      <HierarchicalDemo />
      <Note color="info" icon="ti-info-circle">
        The threshold slider is deliberately log-scaled rather than linear. Ward's merge cost
        scales with cluster size, so late merges of already-separated clusters can be an order of
        magnitude more expensive than early ones — a linear slider would spend most of its length
        showing no change at all. Log-scaling spreads that range out evenly instead of hiding it.
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

Z = linkage(X, method="ward")          # the full merge tree
labels = fcluster(Z, t=3, criterion="maxclust")   # cut for exactly 3 clusters

dendrogram(Z)   # matplotlib visualization`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="What's the practical advantage of hierarchical clustering over K-Means, given that it's slower?"
          a="It doesn't require committing to a number of clusters before fitting — the full dendrogram lets you inspect groupings at every level of granularity and pick a cut afterward, or even use different cuts for different purposes. It's also deterministic (no random initialization to worry about) and can capture nested cluster structure that a single flat K-Means partition can't represent at all." />
      <QA q="Why does single linkage tend to 'chain,' and is that always undesirable?"
          a="Single linkage only cares about the single closest pair of points between two clusters, so a thin bridge of intermediate points can link two otherwise well-separated dense regions one point at a time, producing long, straggly clusters. It's undesirable when the true clusters are compact blobs, but it's actually the right tool when clusters are genuinely elongated or curved, since complete/average/Ward linkage would incorrectly split a long, thin true cluster into pieces." />
      <QA q="Agglomerative clustering is O(n³) in its naive form. Why, and what does that imply for large datasets?"
          a="Each of the n-1 merge steps needs to find the closest pair among the currently remaining clusters, which costs O(n²) per step in the naive approach, giving O(n³) overall — this makes naive hierarchical clustering impractical much past a few thousand points, which is why it's typically reserved for smaller datasets or run on a pre-clustered/sampled subset, unlike K-Means or DBSCAN which scale far more gracefully." />
      <QA q="If you cut a dendrogram built with Ward linkage at a height that produces 3 clusters, is that the same 3-cluster result K-Means would find with K=3?"
          a="Often similar but not guaranteed identical — Ward linkage greedily minimizes variance increase at each individual merge step and can never undo an earlier merge, while K-Means iteratively reassigns points at every centroid update, letting it escape decisions that agglomerative clustering has already locked in. On clearly-separated spherical data they usually agree closely; on ambiguous or overlapping data they can diverge." />
    </div>
  );
}

function SectionDBSCAN() {
  return (
    <div>
      <P>
        DBSCAN takes a completely different starting assumption from K-Means and hierarchical
        clustering: a cluster isn't defined by proximity to a center or by a merge history, it's
        simply a region where points are packed <em>densely</em> together, separated from other
        such regions by sparser space.
      </P>

      <H2 c="Core Concepts" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="Core point">Has at least min_samples points (including itself) within distance epsilon.</Card>
        <Card color="success" title="Border point">Within epsilon of a core point, but doesn't have enough neighbors to be core itself.</Card>
        <Card color="warning" title="Noise point">Neither — too isolated to belong to any dense region.</Card>
      </Grid>
      <P s={{ marginTop: 10 }}>
        Clusters form by chaining together core points that are mutual neighbors, then attaching
        any border points that touch them — no centroid, no merge tree, just connectivity through
        dense regions.
      </P>

      <H2 c="Why It's Different" />
      <Table
        heads={['Property', 'K-Means / Hierarchical', 'DBSCAN']}
        rows={[
          ['Number of clusters', 'Must be chosen in advance (K) or by cutting a tree', 'Discovered automatically'],
          ['Cluster shape', 'Assumes roughly spherical / compact', 'Any shape, including nested crescents or spirals'],
          ['Outliers', 'Every point is forced into some cluster', 'Explicitly labeled as noise, excluded from all clusters'],
        ]}
      />

      <H2 c="Try It — Sweep Epsilon and min_samples" />
      <P>Two interleaving crescents plus scattered noise. This exact shape is why DBSCAN exists — no straight or curved centroid-based boundary handles it cleanly, but density does.</P>
      <DBSCANDemo />
      <Note color="warning" icon="ti-alert-triangle">
        <strong>A real limitation the demo above can't show with just one dataset:</strong> DBSCAN
        struggles when different clusters have genuinely different densities, since a single global
        epsilon can't simultaneously suit a tight cluster and a sparse one — whichever epsilon fits
        the sparse cluster will over-merge the tight one, and vice versa. (HDBSCAN, a newer variant,
        was designed specifically to relax this single-epsilon requirement.)
      </Note>

      <H3 c="A minimal working example" />
      <Code>{`from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X)

# -1 in the output means "noise" — not a real cluster id
n_noise = (labels == -1).sum()`}</Code>

      <H2 c="Interview Q&A" />
      <QA q="Why can DBSCAN correctly separate two interleaving crescent-shaped clusters when K-Means cannot?"
          a="K-Means assigns every point to whichever centroid is nearest in straight-line distance, which only ever produces convex (specifically, roughly spherical) regions — it has no mechanism to follow a curved boundary. DBSCAN instead grows clusters by following chains of locally dense points, so it can trace an arbitrarily curved or elongated shape as long as the points along it stay within epsilon of each other, with no assumption about overall cluster geometry at all." />
      <QA q="What happens to DBSCAN's result if epsilon is set far too large?"
          a="Almost every point ends up within epsilon of enough neighbors to count as core, so the entire dataset (or very large chunks of it) collapses into a single connected cluster — including regions that should have been separate, since a large enough epsilon bridges the sparse gap between them." />
      <QA q="A dataset has one dense cluster and one much sparser one. What goes wrong if you tune epsilon using only the dense cluster?"
          a="An epsilon small enough to correctly resolve the dense cluster's internal structure will likely be too small to connect the sparser cluster's more spread-out points, causing much of it to be misclassified as noise — this is DBSCAN's core limitation with varying-density data, since it uses one global epsilon for the entire dataset." />
      <QA q="Why does DBSCAN not require specifying the number of clusters, unlike K-Means?"
          a="The number of clusters is an emergent property of how many separate dense regions the epsilon/min_samples settings happen to carve out of the data, rather than a parameter fed into the algorithm — this is genuinely useful when the 'right' number of clusters isn't known ahead of time, though it means the cluster count can still shift indirectly if epsilon or min_samples are changed." />
    </div>
  );
}

const SECTION_MAP = {
  kmeans: <SectionKMeans />,
  hierarchical: <SectionHierarchical />,
  dbscan: <SectionDBSCAN />,
};

export default function Clustering() {
  const [active, setActive] = useState('kmeans');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 09</div>
        <h1 className="page-header-title">Clustering</h1>
        <p className="page-header-subtitle">
          Group unlabeled data by similarity, with different assumptions about cluster shape,
          count, and density.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={9} />
    </div>
  );
}
