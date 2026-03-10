export default function TranscriptView({ transcripts }) {
  return (
    <section className="card">
      <h2>Transcript Stream</h2>
      <div className="transcript">
        {transcripts.length === 0 ? <p className="muted">No transcript yet.</p> : null}
        {transcripts.map((item, idx) => (
          <p key={`${item}-${idx}`}>{item}</p>
        ))}
      </div>
    </section>
  );
}
