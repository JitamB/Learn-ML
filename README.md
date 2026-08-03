# Machine Learning Study Platform

A structured, interactive single-page web application for studying Machine Learning — from first
principles through classical algorithms. Built as the companion to the
[Deep Learning Study Platform](https://jitamb.github.io/Deep-Learning-Study-Platform/), reusing the
same design system and navigation shell so the two study guides feel like one continuous experience.

---

## What It Does

- **Hierarchical navigation** — categories in the sidebar collapse/expand; tabs within each module
  switch sub-topics without a page reload
- **Inline math blocks** — monospaced, pre-formatted math notation for equations and derivations
- **Q&A accordions** — interview-style questions with detailed answers, for revision
- **Interactive visualizers** — sliders and live demos embedded per topic (`VizBox`, `SliderRow`)
- **Jupyter-style cells** — available for future hands-on/practical modules (`JupyterCell.jsx`)
- **Light & dark themes** — a shared CSS-token design system, toggled from the top bar

---

## Module Index

| # | Module | Category |
|---|--------|----------|
| 01 | ML Fundamentals | Foundations |

More rows land here as modules are written — keep this table and `src/data/nav.js` in sync.

---

## Project Layout

```
Learn-ML/
├── index.html
├── vite.config.js
├── package.json
└── src/
    ├── main.jsx
    ├── App.jsx
    │
    ├── data/
    │   └── nav.js              # Single source of truth — modules, categories,
    │                           # and helper functions (getModule, getAdjacentModules…)
    │
    ├── styles/
    │   ├── global.css          # Design tokens (colors, fonts, radii)
    │   ├── layout.css          # Sidebar, TopBar, content-shell layout
    │   └── components.css      # Reusable component styles (tables, code blocks,QA, notes, badges, viz-box…)
    │
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.jsx
    │   │   ├── SectionNav.jsx
    │   │   └── NavButtons.jsx
    │   └── ui/
    │       ├── Code.jsx
    │       ├── QA.jsx
    │       ├── Typography.jsx  # Mx (math), H2, H3, P typography helpers
    │       ├── Primitives.jsx  # Note, Table, Grid, Card, VizBox, SliderRow, Badge
    │       └── JupyterCell.jsx
    │
    └── pages/                  # One file per module, lazy-loaded via App.jsx
        ├── _Placeholder.jsx    # "Coming soon" stub for a registered-but-unwritten module
        └── MLFundamentals.jsx
```

---

## Getting Started

**Prerequisites:** Node.js ≥ 18

```bash
npm install
npm run dev
```

Build for production:

```bash
npm run build
npm run preview
```

---

## Adding a New Module

This is the workflow for turning a new batch of content into a page:

1. **Create `src/pages/YourModule.jsx`** (PascalCase filename — it becomes the URL) — follow
   `MLFundamentals.jsx`'s shape:
   - a `TABS` array (`{ id, label, sub }`) for sub-topics
   - one `Section*()` component per tab, built from the shared primitives (`P`, `H2`, `H3`, `Mx`,
     `Table`, `Grid`, `Card`, `Badge`, `Note`, `Code`, `QA`, `VizBox`/`SliderRow`)
   - a `SECTION_MAP` keyed by tab id
   - a default-export page component rendering `page-header` → `SectionNav` → active section →
     `NavButtons`
   - until real content exists, a module can instead render `_Placeholder.jsx` with its
     `moduleId`/`title`/`subtitle`
2. **Register it in `src/data/nav.js`** — add an entry to `MODULES` whose `page` field matches the
   filename exactly (e.g. `page: 'YourModule'` for `YourModule.jsx` — this is what the URL becomes:
   `/module/YourModule`), and make sure its `category` exists in `CATEGORIES` (or add a new one).
3. **Register the lazy import** in `src/App.jsx`'s `PAGE_MAP`, keyed by that same `page` string.

That's it — the sidebar, breadcrumb, prev/next buttons, module counts, and theme all update
automatically since they read from `nav.js`.

### Content guidelines

- Keep each module self-contained in its own page file
- Use the shared primitives rather than one-off inline styles where possible
- Math notation goes in `<Mx block>` blocks — keep it readable in monospace, not LaTeX
- For inline emphasis inside `<P>`, use real JSX children (`<P>text <strong>bold</strong></P>`),
  not an HTML string passed via the `c` prop — `P` renders `c` as plain text, so literal `<strong>`
  tags inside a string prop won't render as bold. `QA`'s `a` prop is the exception: it's rendered
  with `dangerouslySetInnerHTML`, so HTML strings work there.
- Test both light and dark mode before considering a module done

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI Framework | React 18 |
| Routing | React Router v6 |
| Bundler | Vite 5 |
| Styling | Vanilla CSS (custom design tokens) |
| Icons | Tabler Icons (CDN) |
| Fonts | Inter (sans), JetBrains Mono (code) |

No external UI library. No Tailwind. Pure CSS design system with custom tokens for both light and
dark modes — the same tokens as the Deep Learning platform, so both sites feel consistent.

---

## Deployment

Pushes to `main` build via Vite and publish `dist/` to GitHub Pages through
`.github/workflows/deploy.yml` (same pattern as the Deep Learning platform). `vite.config.js`'s
`base` defaults to `/Learn-ML/` but reads a `DEPLOY_BASE` env var first, and `src/main.jsx` derives
its router `basePath` from Vite's `import.meta.env.BASE_URL` — so building with
`DEPLOY_BASE=/some-other-path/ npm run build` retargets both without editing code. This is what
lets the same repo be built once for `jitamb.github.io/Learn-ML/` and again for inclusion under
[study.jitam.in](https://study.jitam.in). Update `index.html`'s OG/Twitter URLs by hand if the
canonical deployment path changes.

---

> Companion to the [Deep Learning Study Platform](https://jitamb.github.io/Deep-Learning-Study-Platform/) — same design system, classical-ML content.

---

## Credits & Sources

- **_Jitam Barman_** — curation, content, and platform design
- **_Aurélien Géron_**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (O'Reilly Media) — primary reference for module content
- **_CampusX_** - 100 Days of Machine Learning Playlist
- **_Claude_** (Anthropic) — AI pair-programmer for this platform
- **_IIT Kharagpur_**
