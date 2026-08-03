/** SectionNav — horizontal tab bar for within-module section navigation */
export default function SectionNav({ tabs, active, onSelect, onChange }) {
  const handleSelect = onSelect ?? onChange;
  return (
    <div className="section-tabs" role="tablist">
      {tabs.map(({ id, label, sub }) => (
        <button
          key={id}
          role="tab"
          aria-selected={active === id}
          className={`section-tab${active === id ? ' active' : ''}`}
          onClick={() => handleSelect(id)}
          id={`tab-${id}`}
          aria-controls={`panel-${id}`}
        >
          {label}
          {sub && <div className="section-tab-sub">{sub}</div>}
        </button>
      ))}
    </div>
  );
}
