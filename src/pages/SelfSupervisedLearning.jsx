import { useState, useMemo, useRef } from 'react';
import { Mx, H2, H3, P } from '../components/ui/Typography.jsx';
import Code from '../components/ui/Code.jsx';
import QA from '../components/ui/QA.jsx';
import { Note, Table, Grid, Card, VizBox, SliderRow } from '../components/ui/Primitives.jsx';
import SectionNav from '../components/layout/SectionNav.jsx';
import NavButtons from '../components/layout/NavButtons.jsx';

/* ── Shared helpers ─────────────────────────────────────────── */
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
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
function zerosVec(n) { return new Array(n).fill(0); }
function randMatrix(rows, cols, rand, scale) { return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (rand() * 2 - 1) * scale)); }
function matVec(W, v) { return W.map(row => row.reduce((s, w, i) => s + w * v[i], 0)); }
function addVec(a, b) { return a.map((x, i) => x + b[i]); }
function tanhVec(v) { return v.map(Math.tanh); }
function sigmoidVec(v) { return v.map(sigmoid); }

/* ── Shared bitmap glyphs (used by both Autoencoder & Contrastive tabs) ── */
const GLYPH_NAMES = ['ring', 'stroke', 'seven', 'cross', 'plus', 'square', 'diagonal', 'bars'];
const GLYPH_RAW = {
  ring:     ['011110', '100001', '100001', '100001', '100001', '011110'],
  stroke:   ['001100', '011100', '001100', '001100', '001100', '011110'],
  seven:    ['111111', '000010', '000100', '001000', '010000', '010000'],
  cross:    ['100001', '010010', '001100', '001100', '010010', '100001'],
  plus:     ['000000', '001100', '001100', '111111', '001100', '001100'],
  square:   ['111110', '100010', '100010', '100010', '100010', '111110'],
  diagonal: ['000001', '000010', '000100', '001000', '010000', '100000'],
  bars:     ['011010', '011010', '011010', '011010', '011010', '011010'],
};
const GRID = 6, DIM = 36;
function flattenGlyph(rows) { return rows.join('').split('').map(Number); }
const GLYPHS = GLYPH_NAMES.map(n => flattenGlyph(GLYPH_RAW[n]));

/* ── Autoencoder: real nonlinear net (36→8 tanh→2 linear→8 tanh→36 sigmoid) ── */
function aeInitNet(seed, H) {
  const rand = seededRandom(seed);
  return {
    W1: randMatrix(H, DIM, rand, 0.4), b1: zerosVec(H),
    W2: randMatrix(2, H, rand, 0.4), b2: zerosVec(2),
    W3: randMatrix(H, 2, rand, 0.4), b3: zerosVec(H),
    W4: randMatrix(DIM, H, rand, 0.4), b4: zerosVec(DIM),
  };
}
function aeForward(net, x) {
  const a1 = tanhVec(addVec(matVec(net.W1, x), net.b1));
  const z2 = addVec(matVec(net.W2, a1), net.b2);
  const a3 = tanhVec(addVec(matVec(net.W3, z2), net.b3));
  const yhat = sigmoidVec(addVec(matVec(net.W4, a3), net.b4));
  return { a1, z2, a3, yhat };
}
function aeDecode(net, z2) {
  const a3 = tanhVec(addVec(matVec(net.W3, z2), net.b3));
  return sigmoidVec(addVec(matVec(net.W4, a3), net.b4));
}
function aeTrainStep(net, samples, lr) {
  const H = net.b1.length;
  const gW1 = randMatrix(H, DIM, () => 0, 0), gb1 = zerosVec(H);
  const gW2 = randMatrix(2, H, () => 0, 0), gb2 = zerosVec(2);
  const gW3 = randMatrix(H, 2, () => 0, 0), gb3 = zerosVec(H);
  const gW4 = randMatrix(DIM, H, () => 0, 0), gb4 = zerosVec(DIM);
  for (const x of samples) {
    const { a1, z2, a3, yhat } = aeForward(net, x);
    const delta4 = yhat.map((y, i) => (2 / DIM) * (y - x[i]) * y * (1 - y));
    for (let i = 0; i < DIM; i++) { gb4[i] += delta4[i]; for (let j = 0; j < H; j++) gW4[i][j] += delta4[i] * a3[j]; }
    const da3 = Array.from({ length: H }, (_, j) => delta4.reduce((s, d, i) => s + d * net.W4[i][j], 0));
    const dz3 = da3.map((d, j) => d * (1 - a3[j] * a3[j]));
    for (let i = 0; i < H; i++) { gb3[i] += dz3[i]; for (let j = 0; j < 2; j++) gW3[i][j] += dz3[i] * z2[j]; }
    const dz2 = Array.from({ length: 2 }, (_, j) => dz3.reduce((s, d, i) => s + d * net.W3[i][j], 0));
    for (let i = 0; i < 2; i++) { gb2[i] += dz2[i]; for (let j = 0; j < H; j++) gW2[i][j] += dz2[i] * a1[j]; }
    const da1 = Array.from({ length: H }, (_, j) => dz2.reduce((s, d, i) => s + d * net.W2[i][j], 0));
    const dz1 = da1.map((d, j) => d * (1 - a1[j] * a1[j]));
    for (let i = 0; i < H; i++) { gb1[i] += dz1[i]; for (let j = 0; j < DIM; j++) gW1[i][j] += dz1[i] * x[j]; }
  }
  const n = samples.length;
  for (let i = 0; i < H; i++) { net.b1[i] -= lr * gb1[i] / n; for (let j = 0; j < DIM; j++) net.W1[i][j] -= lr * gW1[i][j] / n; }
  for (let i = 0; i < 2; i++) { net.b2[i] -= lr * gb2[i] / n; for (let j = 0; j < H; j++) net.W2[i][j] -= lr * gW2[i][j] / n; }
  for (let i = 0; i < H; i++) { net.b3[i] -= lr * gb3[i] / n; for (let j = 0; j < 2; j++) net.W3[i][j] -= lr * gW3[i][j] / n; }
  for (let i = 0; i < DIM; i++) { net.b4[i] -= lr * gb4[i] / n; for (let j = 0; j < H; j++) net.W4[i][j] -= lr * gW4[i][j] / n; }
}
function trainAutoencoder() {
  const net = aeInitNet(17, 8);
  for (let it = 0; it < 6000; it++) aeTrainStep(net, GLYPHS, 0.5);
  return net;
}
// Trained once when this module loads (fixed data, fixed hyperparameters — always the same
// result), not on every mount of AutoencoderDemo, so switching away from and back to this tab
// doesn't pay the ~6000-iteration training cost again.
const AE_NET = trainAutoencoder();
const AE_PRESET_LATENTS = GLYPHS.map(p => aeForward(AE_NET, p).z2);

const AE_DOMAIN = 9, AE_SVG = 260;
const aeClamp = v => Math.max(-AE_DOMAIN, Math.min(AE_DOMAIN, v));
const aeXToPx = x => (x + AE_DOMAIN) / (2 * AE_DOMAIN) * AE_SVG;
const aeYToPx = y => (1 - (y + AE_DOMAIN) / (2 * AE_DOMAIN)) * AE_SVG;
const aePxToX = px => (px / AE_SVG) * (2 * AE_DOMAIN) - AE_DOMAIN;
const aePxToY = py => (1 - py / AE_SVG) * (2 * AE_DOMAIN) - AE_DOMAIN;

function PixelGrid({ values, onToggle, shade }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${GRID}, 24px)`, gap: 2 }}>
      {values.map((v, i) => (
        <div key={i} onClick={onToggle ? () => onToggle(i) : undefined} style={{
          width: 24, height: 24, borderRadius: 3, cursor: onToggle ? 'pointer' : 'default',
          background: 'var(--color-text-primary)', opacity: shade ? v : (v ? 1 : 0.06),
          border: '1px solid var(--color-border-tertiary)',
        }} />
      ))}
    </div>
  );
}

function AutoencoderDemo() {
  const svgRef = useRef(null);
  const [draw, setDraw] = useState(() => GLYPHS[0].slice());
  const [manualLatent, setManualLatent] = useState(null);
  const [dragging, setDragging] = useState(false);

  const encodedLatent = useMemo(() => aeForward(AE_NET, draw).z2, [draw]);
  const latent = manualLatent ?? encodedLatent;
  const recon = useMemo(() => aeDecode(AE_NET, latent), [latent]);

  function togglePixel(i) {
    setDraw(prev => { const copy = prev.slice(); copy[i] = copy[i] ? 0 : 1; return copy; });
    setManualLatent(null);
  }
  function loadPreset(idx) { setDraw(GLYPHS[idx].slice()); setManualLatent(null); }
  function clearDraw() { setDraw(new Array(DIM).fill(0)); setManualLatent(null); }

  function handlePointerMove(e) {
    if (!dragging || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const scale = AE_SVG / rect.width;
    const px = (e.clientX - rect.left) * scale, py = (e.clientY - rect.top) * scale;
    setManualLatent([aeClamp(aePxToX(px)), aeClamp(aePxToY(py))]);
  }
  function startDrag(e) { e.target.setPointerCapture(e.pointerId); setDragging(true); }
  function endDrag() { setDragging(false); }

  return (
    <VizBox>
      <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Draw (click to toggle)</div>
          <PixelGrid values={draw} onToggle={togglePixel} />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8, maxWidth: 174 }}>
            {GLYPH_NAMES.map((name, idx) => (
              <button key={name} onClick={() => loadPreset(idx)} style={{ ...toggleBtnStyle(false), padding: '3px 7px', fontSize: 10.5 }}>{name}</button>
            ))}
            <button onClick={clearDraw} style={{ ...toggleBtnStyle(false), padding: '3px 7px', fontSize: 10.5 }}>clear</button>
          </div>
        </div>

        <svg ref={svgRef} viewBox={`0 0 ${AE_SVG} ${AE_SVG}`} width={AE_SVG} height={AE_SVG}
          style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)', touchAction: 'none' }}
          onPointerMove={handlePointerMove} onPointerUp={endDrag}>
          {AE_PRESET_LATENTS.map((z, idx) => (
            <g key={idx}>
              <circle cx={aeXToPx(z[0])} cy={aeYToPx(z[1])} r={4} fill="var(--color-border-tertiary)" />
              <text x={aeXToPx(z[0]) + 6} y={aeYToPx(z[1]) + 3} fontSize={9} fill="var(--color-text-tertiary)">{GLYPH_NAMES[idx]}</text>
            </g>
          ))}
          <circle cx={aeXToPx(latent[0])} cy={aeYToPx(latent[1])} r={7}
            fill="var(--color-background-info)" stroke="var(--color-border-info)" strokeWidth={2}
            style={{ cursor: 'grab' }} onPointerDown={startDrag} />
        </svg>

        <div>
          <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginBottom: 6 }}>Reconstruction</div>
          <PixelGrid values={recon} shade />
        </div>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 12 }}>
        Toggle pixels to draw your own input — it encodes live to the blue dot on the latent plot.
        Or grab the blue dot directly and drag it anywhere; the reconstruction on the right morphs
        continuously, decoded fresh from wherever the dot currently sits.
      </div>
    </VizBox>
  );
}

/* ── Contrastive Learning: real encoder trained with a pairwise hinge loss ── */
function ceInitNet(seed, H) {
  const rand = seededRandom(seed);
  return { W1: randMatrix(H, DIM, rand, 0.4), b1: zerosVec(H), W2: randMatrix(2, H, rand, 0.4), b2: zerosVec(2) };
}
function ceEmbed(net, x) {
  const a1 = tanhVec(addVec(matVec(net.W1, x), net.b1));
  return { a1, z2: addVec(matVec(net.W2, a1), net.b2) };
}
function flipH(x) {
  const out = new Array(DIM);
  for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) out[r * GRID + c] = x[r * GRID + (GRID - 1 - c)];
  return out;
}
function shiftRight(x, amt) {
  if (!amt) return x.slice();
  const out = new Array(DIM);
  for (let r = 0; r < GRID; r++) for (let c = 0; c < GRID; c++) out[r * GRID + c] = x[r * GRID + ((c - amt + GRID) % GRID)];
  return out;
}
function noiseFlip(x, indices) { const out = x.slice(); indices.forEach(i => { out[i] = 1 - out[i]; }); return out; }
function noisePermutation(seed) {
  const rand = seededRandom(seed);
  const idx = Array.from({ length: DIM }, (_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1));[idx[i], idx[j]] = [idx[j], idx[i]]; }
  return idx;
}
function applyAug(base, perm, { flip = false, shift = 0, noiseK = 0 }) {
  let x = flip ? flipH(base) : base.slice();
  x = shiftRight(x, shift);
  return noiseFlip(x, perm.slice(0, noiseK));
}
const CE_PERMS = GLYPH_NAMES.map((_, b) => noisePermutation(500 + b));
const CE_RECIPES = [
  { flip: false, shift: 0, noiseK: 0 }, { flip: true, shift: 0, noiseK: 0 },
  { flip: false, shift: 1, noiseK: 0 }, { flip: false, shift: 2, noiseK: 0 }, { flip: false, shift: 3, noiseK: 0 },
  { flip: false, shift: 0, noiseK: 2 }, { flip: false, shift: 0, noiseK: 4 },
  { flip: false, shift: 0, noiseK: 6 }, { flip: false, shift: 0, noiseK: 8 },
  { flip: true, shift: 1, noiseK: 0 }, { flip: false, shift: 1, noiseK: 4 }, { flip: true, shift: 1, noiseK: 4 },
];
const CE_TRAIN = [];
GLYPHS.forEach((base, b) => { CE_RECIPES.forEach(recipe => CE_TRAIN.push({ x: applyAug(base, CE_PERMS[b], recipe), base: b })); });

function ceTrainStep(net, samples, margin, lr) {
  const H = net.b1.length, n = samples.length;
  const embeds = samples.map(s => ceEmbed(net, s.x));
  const gW1 = randMatrix(H, DIM, () => 0, 0), gb1 = zerosVec(H);
  const gW2 = randMatrix(2, H, () => 0, 0), gb2 = zerosVec(2);
  function backprop(i, j, dzi, dzj) {
    for (const [idx, dz] of [[i, dzi], [j, dzj]]) {
      const a1 = embeds[idx].a1, x = samples[idx].x;
      for (let k = 0; k < 2; k++) { gb2[k] += dz[k]; for (let h = 0; h < H; h++) gW2[k][h] += dz[k] * a1[h]; }
      const da1 = Array.from({ length: H }, (_, h) => dz.reduce((s, d, k) => s + d * net.W2[k][h], 0));
      const dz1 = da1.map((d, h) => d * (1 - a1[h] * a1[h]));
      for (let h = 0; h < H; h++) { gb1[h] += dz1[h]; for (let d = 0; d < DIM; d++) gW1[h][d] += dz1[h] * x[d]; }
    }
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const zi = embeds[i].z2, zj = embeds[j].z2;
    const diff = [zi[0] - zj[0], zi[1] - zj[1]];
    const d = Math.hypot(diff[0], diff[1]);
    if (samples[i].base === samples[j].base) backprop(i, j, [2 * diff[0], 2 * diff[1]], [-2 * diff[0], -2 * diff[1]]);
    else if (d < margin) { const coeff = -2 * (margin - d) / (d + 1e-8); backprop(i, j, [coeff * diff[0], coeff * diff[1]], [-coeff * diff[0], -coeff * diff[1]]); }
  }
  const totalPairs = n * (n - 1) / 2;
  for (let h = 0; h < H; h++) { net.b1[h] -= lr * gb1[h] / totalPairs; for (let d = 0; d < DIM; d++) net.W1[h][d] -= lr * gW1[h][d] / totalPairs; }
  for (let k = 0; k < 2; k++) { net.b2[k] -= lr * gb2[k] / totalPairs; for (let h = 0; h < H; h++) net.W2[k][h] -= lr * gW2[k][h] / totalPairs; }
}
function trainContrastive() {
  const net = ceInitNet(11, 8);
  for (let it = 0; it < 300; it++) ceTrainStep(net, CE_TRAIN, 6, 0.3);
  return net;
}
// Trained once when this module loads, same reasoning as AE_NET above — this is the demo
// that's most expensive to retrain (300 iterations x ~4,560 pairwise comparisons each).
const CE_NET = trainContrastive();
const CE_TRAIN_EMBEDS = CE_TRAIN.map(s => ({ base: s.base, z: ceEmbed(CE_NET, s.x).z2 }));
const CE_CENTROIDS = GLYPH_NAMES.map((_, b) => {
  const pts = CE_TRAIN_EMBEDS.filter(e => e.base === b);
  return [pts.reduce((s, e) => s + e.z[0], 0) / pts.length, pts.reduce((s, e) => s + e.z[1], 0) / pts.length];
});

function ContrastiveDemo() {
  const [baseIdx, setBaseIdx] = useState(0);
  const [flip, setFlip] = useState(false);
  const [shift, setShift] = useState(0);
  const [noiseK, setNoiseK] = useState(0);

  const probeX = useMemo(() => applyAug(GLYPHS[baseIdx], CE_PERMS[baseIdx], { flip, shift, noiseK }), [baseIdx, flip, shift, noiseK]);
  const probeZ = useMemo(() => ceEmbed(CE_NET, probeX).z2, [probeX]);
  const nearestBase = useMemo(() => {
    let best = -1, bestD = Infinity;
    CE_TRAIN_EMBEDS.forEach(e => { const d = Math.hypot(e.z[0] - probeZ[0], e.z[1] - probeZ[1]); if (d < bestD) { bestD = d; best = e.base; } });
    return best;
  }, [probeZ]);

  const xs = CE_TRAIN_EMBEDS.map(e => e.z[0]), ys = CE_TRAIN_EMBEDS.map(e => e.z[1]);
  const domain = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2 + 2;
  const cx0 = (Math.max(...xs) + Math.min(...xs)) / 2, cy0 = (Math.max(...ys) + Math.min(...ys)) / 2;
  const SVGN = 280;
  const xToPx = x => (x - (cx0 - domain)) / (2 * domain) * SVGN;
  const yToPx = y => (1 - (y - (cy0 - domain)) / (2 * domain)) * SVGN;

  return (
    <VizBox>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
        {GLYPH_NAMES.map((name, idx) => (
          <button key={name} onClick={() => setBaseIdx(idx)} style={toggleBtnStyle(idx === baseIdx)}>{name}</button>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <div style={{ minWidth: 170 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12.5, marginBottom: 12 }}>
            <input type="checkbox" checked={flip} onChange={e => setFlip(e.target.checked)} /> Flip
          </label>
          <SliderRow label="Shift" min={0} max={3} step={1} value={shift} onChange={setShift} fmt={v => String(v)} />
          <SliderRow label="Noise" min={0} max={8} step={1} value={noiseK} onChange={setNoiseK} fmt={v => String(v)} />
          <div style={{ fontSize: 12, marginTop: 12, color: 'var(--color-text-secondary)' }}>
            Currently nearest to:{' '}
            <strong style={{ color: nearestBase === baseIdx ? 'var(--color-text-success)' : 'var(--color-text-danger)' }}>
              {GLYPH_NAMES[nearestBase]}
            </strong>
          </div>
        </div>
        <svg viewBox={`0 0 ${SVGN} ${SVGN}`} width={SVGN} height={SVGN} style={{ background: 'var(--color-background-secondary)', borderRadius: 'var(--border-radius-md)' }}>
          {CE_TRAIN_EMBEDS.map((e, idx) => (
            <circle key={idx} cx={xToPx(e.z[0])} cy={yToPx(e.z[1])} r={e.base === baseIdx ? 3.5 : 2.5}
              fill={e.base === baseIdx ? 'var(--color-background-info)' : 'var(--color-border-tertiary)'}
              opacity={e.base === baseIdx ? 0.9 : 0.5} />
          ))}
          <line x1={xToPx(probeZ[0])} y1={yToPx(probeZ[1])} x2={xToPx(CE_CENTROIDS[baseIdx][0])} y2={yToPx(CE_CENTROIDS[baseIdx][1])}
            stroke="var(--color-border-info)" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.8} />
          <circle cx={xToPx(probeZ[0])} cy={yToPx(probeZ[1])} r={6} fill="var(--color-background-danger)" stroke="var(--color-border-danger)" strokeWidth={2} />
        </svg>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--color-text-tertiary)', marginTop: 12 }}>
        Small blue dots are the 12 trained augmented views of "{GLYPH_NAMES[baseIdx]}"; gray dots are
        every other glyph's trained views. The red dot is this exact augmented view, embedded live —
        the dashed line points back to its family's center. Move one control at a time and it stays
        anchored; stack Flip, a large Shift, and heavy Noise together, and watch it occasionally drift.
      </div>
    </VizBox>
  );
}

/* ── Tabs ───────────────────────────────────────────────────── */
const TABS = [
  { id: 'pretext', label: 'Pretext Tasks', sub: 'Manufacturing labels out of the data itself' },
  { id: 'autoencoder', label: 'Autoencoders', sub: 'Compress, then reconstruct' },
  { id: 'contrastive', label: 'Contrastive Learning', sub: 'Learning by comparison' },
];

function SectionPretext() {
  return (
    <div>
      <P>
        Self-supervised learning takes the "generate your own labels" idea to its logical extreme:
        instead of a small labeled pool at all, it manufactures a supervised training signal entirely
        out of the raw, unlabeled data's own structure — hide or scramble some part of an input and
        train a model to recover it. The trick is designing a "pretext task" hard enough that solving
        it forces the model to learn genuinely useful structure, not something it can shortcut.
      </P>

      <H2 c="Vision Pretext Tasks" />
      <Grid cols={3} gap={10}>
        <Card color="info" title="Rotation Prediction">Rotate an image 0°/90°/180°/270° and train the model to guess which. Getting this right requires understanding what the object normally looks like right-side up.</Card>
        <Card color="success" title="Colorization">Strip the color from an image and train the model to predict it back. Requires learning what real-world objects and textures typically look like.</Card>
        <Card color="warning" title="Jigsaw Puzzles">Cut an image into patches, shuffle them, and train the model to predict the correct arrangement — forces it to learn spatial relationships between parts of a scene.</Card>
      </Grid>

      <H2 c="Language Pretext Tasks" />
      <P>
        <strong>Masked Language Modeling</strong> hides a random word (or token) in a sentence and
        trains the model to predict it from context in both directions. This one pretext task, scaled
        up to enormous datasets and model sizes, is the foundation nearly every modern large language
        model's pretraining is built on.
      </P>
      <Note color="info" icon="ti-info-circle">
        For how masked-token pretraining scales up to full language models, see "Transformers Part 2"
        on the Deep Learning Study Platform.
      </Note>

      <H2 c="What Each Pretext Task Actually Forces the Model to Learn" />
      <Table
        heads={['Pretext Task', 'Input Manipulation', 'What Solving It Requires Learning']}
        rows={[
          ['Rotation prediction', 'Rotate by a random multiple of 90°', 'Canonical object orientation and structure'],
          ['Colorization', 'Strip color channels', 'Real-world texture and object-color association'],
          ['Jigsaw puzzles', 'Cut into patches, shuffle', 'Spatial layout and part-whole relationships'],
          ['Masked language modeling', 'Hide random tokens', 'Syntax, semantics, and long-range context'],
        ]}
      />
      <P>
        None of these tasks are interesting on their own — nobody actually needs a rotation classifier.
        The point is always the same: throw away the pretext task's output head after pretraining, and
        keep the learned internal representations, which transfer surprisingly well to whatever
        downstream task you actually care about (classification, detection, etc.) — often with far
        less labeled data than training from scratch would need.
      </P>

      <H2 c="Interview Q&A" />
      <QA q="What makes a pretext task 'good' for self-supervised learning?"
          a="It has to require genuinely understanding the data's real structure to solve well, and it has to be impossible (or at least hard) to shortcut with a trivial heuristic — a task the model can solve by memorizing low-level statistics without learning anything transferable defeats the purpose." />
      <QA q="Why bother with a pretext task at all instead of just labeling more data?"
          a="Unlabeled data is usually vastly more abundant and cheaper to obtain than labeled data — pretext tasks let a model learn useful general representations from that abundant supply, so the (still necessary) labeled data only has to fine-tune a smaller downstream task, not teach the model everything about the domain from scratch." />
      <QA q="After pretraining on a pretext task, how does the resulting model actually get used?"
          a="The pretext task's final output layer (e.g. the rotation classifier's 4-way softmax) is discarded, and the learned encoder — everything before that head — is kept and fine-tuned (or used as fixed features) on the actual downstream task, which usually needs far fewer labeled examples to reach good performance than training that same architecture from a random initialization." />
      <QA q="Is masked language modeling considered self-supervised or unsupervised?"
          a="Self-supervised — the label (the masked-out token) comes directly and automatically from the input itself, which is the defining trait of self-supervised learning, as opposed to unsupervised learning's total absence of any target signal (e.g. clustering, which never predicts anything, just groups)." />
    </div>
  );
}

function SectionAutoencoder() {
  return (
    <div>
      <P>
        An autoencoder is an hourglass-shaped network: squeeze the input through a narrow bottleneck,
        then try to rebuild the original from just what made it through. If the reconstruction comes
        out close to the original, the bottleneck must have preserved whatever information actually
        mattered — that compressed bottleneck representation is the whole point, not the reconstruction
        itself.
      </P>

      <H2 c="Encoder, Bottleneck, Decoder" />
      <Mx block>{`  encoder:  x  →  z   (compress: high-dimensional input → low-dimensional code)
  decoder:  z  →  x̂   (reconstruct: low-dimensional code → back to input's shape)

  loss = ‖x − x̂‖²   (reconstruction error — minimize it, with z forced through
                      a bottleneck too narrow to just memorize x directly)`}</Mx>
      <Note color="info" icon="ti-info-circle">
        A <em>linear</em> encoder and decoder trained on exactly this squared-error loss converges to
        the same subspace PCA finds (see Dimensionality Reduction) — same math, different name. The
        demo below uses a genuinely <em>nonlinear</em> encoder/decoder (a small hidden layer with a
        tanh activation on each side), which is where autoencoders actually earn their keep over PCA:
        they can capture curved, non-linear structure a straight-line projection never could.
      </Note>

      <H2 c="Try It — Draw, Encode, Drag, Reconstruct" />
      <P>
        A real 2-layer encoder and decoder, trained by hand-derived backpropagation on 8 small 6×6
        bitmap glyphs until reconstruction is essentially perfect. Draw your own input on the left —
        it encodes live to a point on the 2D latent map. Grab that point directly and drag it anywhere
        to watch the decoder reconstruct whatever lands there, continuously.
      </P>
      <AutoencoderDemo />
      <P s={{ marginTop: 10 }}>
        Try dragging from one glyph's dot to another's — the reconstruction morphs smoothly through
        the space between them, evidence the latent space is a genuinely organized 2D surface, not
        just 8 memorized lookup points with nothing meaningful in between.
      </P>

      <H3 c="A minimal working example" />
      <Code>{`import torch, torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, in_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, in_dim), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

model = Autoencoder()
loss_fn = nn.MSELoss()
loss = loss_fn(model(x_batch), x_batch)   # target IS the input`}</Code>

      <Note color="warning" icon="ti-alert-triangle">
        Reconstruction quality alone doesn't guarantee the latent space is <em>useful</em> for
        anything else. Nothing in the loss rewards pulling semantically different inputs apart or
        pushing similar ones together — it only rewards rebuilding each input from its own code. Two
        inputs that are visually similar but semantically unrelated can end up with nearby codes just
        because they happen to compress similarly, which is exactly the gap contrastive learning
        (next tab) is designed to close.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Why does the bottleneck need to be narrower than the input, rather than just training a network to output a copy of its input?"
          a="Without a bottleneck (or some other constraint), the easiest way to minimize reconstruction error is the identity function — copy the input straight through, learning nothing about its structure. Forcing the information through a narrower layer than the input means the network can only reconstruct well if the bottleneck captures the input's genuinely important structure, discarding redundancy and noise." />
      <QA q="What's the relationship between a linear autoencoder and PCA?"
          a="A linear encoder and decoder trained to minimize squared reconstruction error converges to the same subspace that PCA's top principal components span — both are solving the same 'best low-rank linear approximation' problem, just arrived at via gradient descent instead of eigendecomposition. Autoencoders only diverge meaningfully from PCA once nonlinear activations are introduced." />
      <QA q="Give an example of how an autoencoder's learned features could be reused for a downstream task."
          a="Train the autoencoder on a large pool of unlabeled images, then discard the decoder entirely and treat the encoder's bottleneck output as a compact feature vector for each image — those features (learned with zero labels) can then feed a much smaller supervised classifier trained on a small labeled subset, often performing better than training that same classifier directly on raw pixels." />
      <QA q="What's a 'denoising autoencoder,' and why might it learn more useful features than a plain one?"
          a="It deliberately corrupts the input (e.g. adds noise, or masks patches) before encoding, but still trains against the clean, uncorrupted original as the reconstruction target. This forces the network to learn robust structure that generalizes past exact pixel values, rather than potentially learning a degenerate near-identity mapping when the bottleneck alone isn't a tight enough constraint." />
    </div>
  );
}

function SectionContrastive() {
  return (
    <div>
      <P>
        Contrastive learning takes the "magnet" metaphor and makes it literal: instead of asking a
        network to reconstruct anything, train it purely by comparison — pull the embeddings of
        things that should be considered "the same" closer together in vector space, and push the
        embeddings of things that are genuinely different further apart.
      </P>

      <H2 c="Positive and Negative Pairs" />
      <P>
        A <strong>positive pair</strong> is two different augmented views of the same underlying item
        (for real images: crop, blur, color jitter of one photo; here: flip, shift, or pixel-noise of
        one bitmap glyph). A <strong>negative pair</strong> is views coming from two different items.
        No human ever labels anything "same" or "different" — that structure comes entirely from
        knowing which augmented views were generated from which original, which is why this is
        self-supervised rather than supervised.
      </P>
      <Mx block>{`  Pairwise contrastive loss (implemented in the demo below):

  positive pair (i, j):   loss = d(i, j)²                       (pull together)
  negative pair (i, k):   loss = max(0, margin − d(i, k))²      (push apart, up to a margin)

  InfoNCE / NT-Xent (what SimCLR and MoCo actually use — described here, not implemented):

  loss_i = −log(  exp(sim(i, pos(i)) / τ)  /  Σ_{k≠i} exp(sim(i, k) / τ)  )

  sim = cosine similarity, τ = a temperature hyperparameter. It's a softmax over
  similarity to every other sample in the batch, rather than a hand-picked margin —
  more expensive to backprop through (cosine similarity plus a full-batch softmax),
  which is why the from-scratch demo below uses the simpler, original pairwise
  hinge form instead.`}</Mx>

      <H2 c="Try It — Pull Toward Family, Push Away From Strangers" />
      <P>
        The same 8 bitmap glyphs from the Autoencoders tab, now embedded by a small encoder trained
        with the pairwise contrastive loss above — no decoder this time, just an embedding space
        shaped entirely by which augmented views came from the same source.
      </P>
      <ContrastiveDemo />

      <H3 c="A minimal working example" />
      <Code>{`from pytorch_metric_learning.losses import NTXentLoss

loss_fn = NTXentLoss(temperature=0.5)   # the real InfoNCE / NT-Xent loss

# embeddings from two augmented views of the same batch of images
z1, z2 = encoder(augment(x)), encoder(augment(x))
embeddings = torch.cat([z1, z2])
labels = torch.arange(len(x)).repeat(2)   # view i and view i+N are a positive pair
loss = loss_fn(embeddings, labels)`}</Code>

      <Note color="warning" icon="ti-alert-triangle">
        Moving <em>one</em> augmentation control at a time, the probe reliably lands back in its own
        family — every tested shift and noise level, for every glyph. Stack several augmentations at
        their extremes <em>simultaneously</em>, though, and it can drift into the wrong family's
        territory. That's a real, honest property of learned invariance, not a bug in this demo:
        contrastive encoders generalize well within the range and combinations of augmentations they
        were actually trained against, and less predictably outside it — which is exactly why real
        SimCLR-style pipelines train against a large, deliberately diverse augmentation set.
      </Note>

      <H2 c="Interview Q&A" />
      <QA q="Where do the 'labels' come from in contrastive self-supervised learning, if there's no human annotation at all?"
          a="They come from the data pipeline itself: whichever augmented views were generated from the same original image (or glyph) are automatically treated as a positive pair, and views from different originals are automatically negative pairs. The supervision signal is 'which augmentation produced which view,' which the pipeline already knows without any human labeling it." />
      <QA q="Why do contrastive methods need negative pairs at all — why not just pull all positive pairs together?"
          a="Without negative pairs, the trivial solution that minimizes a pull-together-only loss is collapsing every embedding to the exact same point — perfectly 'similar,' and completely useless. Negative pairs (or an equivalent mechanism, like the momentum-based approaches MoCo pioneered) are what force the embedding space to actually spread out and preserve distinctions between different inputs." />
      <QA q="What is the InfoNCE / NT-Xent loss doing that the simpler pairwise hinge loss implemented in this demo isn't?"
          a="InfoNCE frames each positive pair against every other sample in the batch simultaneously via a softmax, rather than handling each pair's push or pull independently with a fixed margin — this makes the negatives' relative strength adapt automatically (softmax naturally weights harder negatives more) instead of needing a hand-tuned margin, at the cost of needing cosine similarity and a full-batch softmax gradient to implement." />
      <QA q="Why does contrastive learning tend to produce representations that transfer better to downstream tasks than a plain autoencoder's reconstruction-based ones?"
          a="The training objective directly rewards separating different inputs and grouping similar ones in embedding space, which is exactly the property a downstream classifier or retrieval system needs — an autoencoder's reconstruction loss never explicitly asks for that, so any useful separation it produces is an accidental side effect rather than a direct target of training." />
    </div>
  );
}

const SECTION_MAP = {
  pretext: <SectionPretext />,
  autoencoder: <SectionAutoencoder />,
  contrastive: <SectionContrastive />,
};

export default function SelfSupervisedLearning() {
  const [active, setActive] = useState('pretext');
  return (
    <div>
      <div className="page-header">
        <div className="page-header-eyebrow">Machine Learning — Module 14</div>
        <h1 className="page-header-title">Self-Supervised Learning</h1>
        <p className="page-header-subtitle">
          Generating target labels from the data itself to learn useful representations with zero
          human labeling.
        </p>
      </div>
      <SectionNav tabs={TABS} active={active} onSelect={setActive} />
      <div role="tabpanel" id={`panel-${active}`} aria-labelledby={`tab-${active}`}>
        {SECTION_MAP[active]}
      </div>
      <NavButtons moduleId={14} />
    </div>
  );
}
