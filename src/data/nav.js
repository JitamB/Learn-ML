/* Central navigation configuration — single source of truth for all modules.
 *
 * This platform grows one module at a time: add an entry to MODULES (its
 * `page` must match the .jsx filename under src/pages/ exactly — it becomes
 * the URL, e.g. page: 'MLFundamentals' -> /module/MLFundamentals), place it
 * under an existing (or new) category in CATEGORIES, register a lazy import
 * in App.jsx's PAGE_MAP keyed by that same `page` string, and create the
 * page file under src/pages/. See the README's "Adding a New Module" section
 * for the full walkthrough.
 */

export const MODULES = [
  { id: 1, page: 'MLFundamentals', title: 'ML Fundamentals', subtitle: 'Definitions, paradigms & workflow', category: 'Foundations' },
];

export const CATEGORIES = [
  { label: 'Foundations', icon: 'ti-brain', moduleIds: [1] },
];

/** Lookup a module by its numeric id */
export function getModule(id) {
  return MODULES.find(m => m.id === id) ?? null;
}

/** Lookup a module by its `page` string (the URL / .jsx filename) */
export function getModuleByPage(page) {
  return MODULES.find(m => m.page === page) ?? null;
}

/** Get the category label string for a given module id */
export function getCategoryForModule(id) {
  const mod = getModule(id);
  return mod?.category ?? '';
}

/** Prev / next module relative to a given id */
export function getAdjacentModules(id) {
  const idx = MODULES.findIndex(m => m.id === id);
  return {
    prev: idx > 0 ? MODULES[idx - 1] : null,
    next: idx < MODULES.length - 1 ? MODULES[idx + 1] : null,
  };
}
