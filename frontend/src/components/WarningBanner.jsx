export default function WarningBanner({ level }) {
  const normalized = (level || "NORMAL").toUpperCase();
  const className =
    normalized === "WARNING"
      ? "banner warning"
      : normalized === "CAUTION"
        ? "banner caution"
        : "banner normal";

  return (
    <section className={className}>
      <h2>{normalized}</h2>
      <p>
        {normalized === "WARNING"
          ? "High risk detected. Stop interaction and verify immediately."
          : normalized === "CAUTION"
            ? "Suspicious signals detected. Verify source before any transfer."
            : "No immediate high-risk signal."}
      </p>
    </section>
  );
}
