/* Central navigation configuration — single source of truth for all modules.
 *
 * This platform grows one module at a time: add an entry to MODULES (its
 * `page` must match the .jsx filename under src/pages/ exactly — it becomes
 * the URL, e.g. page: 'MLFundamentals' -> /MLFundamentals), place it
 * under an existing (or new) category in CATEGORIES, register a lazy import
 * in App.jsx's PAGE_MAP keyed by that same `page` string, and create the
 * page file under src/pages/. See the README's "Adding a New Module" section
 * for the full walkthrough.
 */

export const MODULES = [
  { id: 1, page: 'MLFundamentals', title: 'ML Fundamentals', subtitle: 'Definitions, paradigms & workflow', category: 'Foundations' },
  { id: 2, page: 'EDA', title: 'Exploratory Data Analysis (EDA)', subtitle: 'Getting to know a dataset before modeling it', category: 'Data Handling & Preprocessing' },
  { id: 3, page: 'DataCleaningFeatureEngineering', title: 'Data Cleaning & Feature Engineering', subtitle: 'Missing values, outliers, encoding, scaling & more', category: 'Data Handling & Preprocessing' },
  { id: 4, page: 'LinearModels', title: 'Linear Models', subtitle: 'Linear & Logistic Regression', category: 'Supervised Learning' },
  { id: 5, page: 'KNNDecisionTrees', title: 'KNN & Decision Trees', subtitle: 'Nearest neighbors and recursive rule-based splits', category: 'Supervised Learning' },
  { id: 6, page: 'SVMNaiveBayes', title: 'SVM & Naive Bayes', subtitle: 'Maximum-margin boundaries and Bayesian classification', category: 'Supervised Learning' },
  { id: 7, page: 'BaggingRandomForest', title: 'Bagging & Random Forest', subtitle: 'Combining models to cancel out variance', category: 'Supervised Learning' },
  { id: 8, page: 'Boosting', title: 'Boosting', subtitle: 'Sequential correction — AdaBoost to XGBoost', category: 'Supervised Learning' },
  { id: 9, page: 'Clustering', title: 'Clustering', subtitle: 'k-Means, Hierarchical Clustering & DBSCAN', category: 'Unsupervised Learning' },
  { id: 10, page: 'DimensionalityReduction', title: 'Dimensionality Reduction', subtitle: 'PCA, t-SNE & UMAP', category: 'Unsupervised Learning' },
  { id: 11, page: 'AssociationRulesAnomalyDetection', title: 'Association Rules & Anomaly Detection', subtitle: 'Market-basket mining, Apriori, and flagging what doesn\'t fit', category: 'Unsupervised Learning' },
  { id: 12, page: 'RecommenderSystems', title: 'Recommender Systems', subtitle: 'How "you might also like" actually works', category: 'Unsupervised Learning' },
  { id: 13, page: 'SemiSupervisedLearning', title: 'Semi-Supervised Learning', subtitle: 'Pseudo-Labeling, Self-Training & Label Propagation', category: 'Semi-Supervised & Self-Supervised Learning' },
  { id: 14, page: 'SelfSupervisedLearning', title: 'Self-Supervised Learning', subtitle: 'Pretext Tasks, Autoencoders & Contrastive Learning', category: 'Semi-Supervised & Self-Supervised Learning' },
  { id: 15, page: 'ValidationBiasVariance', title: 'Validation & Bias-Variance', subtitle: 'Train/test/validation splits & cross-validation', category: 'Model Evaluation & Validation' },
  { id: 16, page: 'Metrics', title: 'Metrics', subtitle: 'Classification, Regression & Unsupervised Metrics', category: 'Model Evaluation & Validation' },
  { id: 17, page: 'Regularization', title: 'Regularization', subtitle: 'L1, L2 & Elastic Net', category: 'Model Evaluation & Validation' },
  { id: 18, page: 'HyperparameterTuning', title: 'Hyperparameter Tuning', subtitle: 'Grid Search, Random Search & Bayesian Optimization', category: 'Model Evaluation & Validation' },
  { id: 19, page: 'TimeSeriesAnalysis', title: 'Time Series Fundamentals', subtitle: 'Trend, seasonality, stationarity, ARIMA/SARIMA & exponential smoothing', category: 'Time Series Analysis' },
  { id: 20, page: 'InterpretabilityFairness', title: 'Interpretability & Fairness', subtitle: 'SHAP, LIME, Feature Importance & Fairness Auditing', category: 'Model Interpretability' },
];

export const CATEGORIES = [
  { label: 'Foundations', icon: 'ti-brain', moduleIds: [1] },
  { label: 'Data Handling & Preprocessing', icon: 'ti-database', moduleIds: [2, 3] },
  { label: 'Supervised Learning', icon: 'ti-target', moduleIds: [4, 5, 6, 7, 8] },
  { label: 'Unsupervised Learning', icon: 'ti-affiliate', moduleIds: [9, 10, 11, 12] },
  { label: 'Semi-Supervised & Self-Supervised Learning', icon: 'ti-tags', moduleIds: [13, 14] },
  { label: 'Model Evaluation & Validation', icon: 'ti-gauge', moduleIds: [15, 16, 17, 18] },
  { label: 'Time Series Analysis', icon: 'ti-chart-line', moduleIds: [19] },
  { label: 'Model Interpretability', icon: 'ti-eye', moduleIds: [20] },
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
