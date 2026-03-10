export default function GuidancePanel({ guidance }) {
  const actions = guidance?.actions || [];
  const contacts = guidance?.emergency_contacts || [];

  return (
    <section className="card">
      <h2>Guidance</h2>
      <p className="muted">
        {(guidance?.matched_label || "No category")} {guidance?.summary ? `- ${guidance.summary}` : ""}
      </p>
      <ul>
        {actions.length === 0 ? <li>No recommendation yet.</li> : null}
        {actions.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      {guidance?.banks_notice ? <p className="notice">{guidance.banks_notice}</p> : null}
      <div className="contacts">
        {contacts.map((contact) => (
          <span key={`${contact.name}-${contact.phone}`} className="chip">
            {contact.name}: {contact.phone}
          </span>
        ))}
      </div>
    </section>
  );
}
