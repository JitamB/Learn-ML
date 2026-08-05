import { Link } from 'react-router-dom';
import { H2, P } from '../components/ui/Typography.jsx';
import { Note, Grid, Card, Badge } from '../components/ui/Primitives.jsx';
import { MODULES } from '../data/nav.js';

const CS_FOUNDATIONS = [
  'Python fundamentals (syntax, OOP, functions)',
  'Data structures & algorithms basics',
  'Git/GitHub',
  'Jupyter/Colab environment',
  'NumPy, Pandas',
  'Matplotlib, Seaborn / Plotly',
  'SQL basics',
];

const MATH_FOUNDATIONS=[
  'Linear Algebra: Vectors, matrices, eigenvalues/eigenvectors, SVD',
  'Calculus: Derivatives, gradients',
  'Probability: Distributions, Bayes\' theorem, random variables',
  'Statistics: Hypothesis testing, confidence intervals, MLE/MAP',
  'Optimization: Optimization, gradient descent',
]

// const MATH_FOUNDATIONS = [
//   { topic: 'Linear Algebra', detail: 'Vectors, matrices, eigenvalues/eigenvectors, SVD' },
//   { topic: 'Calculus', detail: 'Derivatives, partial derivatives, gradients' },
//   { topic: 'Probability', detail: "Distributions, Bayes' theorem, random variables" },
//   { topic: 'Statistics', detail: 'Hypothesis testing, confidence intervals, MLE/MAP, sampling' },
//   { topic: 'Optimization', detail: 'Convex optimization, gradient descent, Lagrange multipliers' },
// ];

export default function Home() {
  const firstModule = MODULES[0];

  return (
    <div className="home-page">
    <div className="home-page-inner">
      <div className="page-header">
        <div className="page-header-eyebrow" style={{display: 'flex', justifyContent: 'center'}}>Welcome to</div>
        <h1 className="page-header-title" style={{display: 'flex', justifyContent: 'center'}}>Machine Learning Study Guide</h1>
        <p className="page-header-subtitle" style={{display: 'flex', justifyContent: 'center'}}>
          by Jitam Barman
        </p>
      </div>

      <P>
        Hi, I'm Jitam. This is the guide I wish I'd had when I was first working through classical
        ML. But now it is here for you. 
        It's the companion to my{' '}
        <a href="https://jitamb.github.io/Deep-Learning-Study-Platform/" target="_blank" rel="noreferrer" style={{ textDecoration: 'underline' }}>
          Deep Learning Study Platform
        </a>{' '}
      </P>

      {/* <H2 c="How to use this guide" />
      <Grid cols={2} gap={12}>
        <Card color="info" title="1. Check the prerequisites">
          Skim the two lists below. Shaky on something? Shore it up first — any solid intro resource
          works, this guide assumes it going in rather than teaching it.
        </Card>
        <Card color="success" title="2. Follow the sidebar in order">
          Modules build on each other loosely within a category, but every page is self-contained
          enough to jump to directly if you already know the basics.
        </Card>
        <Card color="purple" title="3. Read, run, then test yourself">
          Each module pairs definitions and math with runnable code, then closes with interview-style
          Q&A so you can check it actually stuck.
        </Card>
        <Card color="warning" title="4. It's a work in progress">
          New modules land over time — the sidebar's module count is always the current total, not
          the finished curriculum.
        </Card>
      </Grid> */}

      {/* <p c="Prerequisites:" /> */}
      <div style={{fontSize: '12pt', paddingTop: '1rem'}}>Before you begin, below are some of the prerequisites to the course</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', alignItems: 'start', paddingTop: '-15rem' }}>
        <div>
          <H2 c="Programming & Tools" />
          <P>Topics this guide assumes going in</P>
          <div style={{ margin: '0.5rem 0 1rem' }}>
            {CS_FOUNDATIONS.map(item => (
              <Badge key={item} color="info">{item}</Badge>
            ))}
          </div>
        </div>

        <div>
          <H2 c="Mathematical Foundations" />
          <P>The math this guide leans on without re-deriving it</P>
          {/* <Grid cols={2} gap={10}>
            {MATH_FOUNDATIONS.map(({ topic, detail }) => (
              <Card key={topic} color="info" title={topic}>{detail}</Card>
            ))}
          </Grid> */}
          <div style={{ margin: '0.5rem 0 1rem' }}>
            {MATH_FOUNDATIONS.map(item => (
              <Badge key={item} color="info">{item}</Badge>
            ))}
          </div>
        </div>
      </div>


      <Note color="success" icon="ti-rocket">
        Comfortable with the above? Good, you're ready to start. If a few gaps remain, that's
        normal too; continue the learning
      </Note>

      {firstModule && (
        <div style={{ textAlign: 'center', marginTop: '2.5rem', paddingTop: '1.5rem', borderTop: '1px solid var(--color-border-tertiary)' }}>
          <Link to={`/${firstModule.page}`} className="home-cta-btn">
            Start with {firstModule.title}
            <i className="ti ti-arrow-right" aria-hidden="true" />
          </Link>
        </div>
      )}
    </div>
    </div>
  );
}
