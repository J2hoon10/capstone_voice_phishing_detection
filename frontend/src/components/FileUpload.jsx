import { useState } from "react";

export default function FileUpload({ onResult }) {
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("audio", file);
      formData.append("threshold", String(threshold));

      const response = await fetch("/api/detect", {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      onResult?.(data);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h2>File Detect</h2>
      <form onSubmit={onSubmit} className="stack">
        <input
          type="file"
          accept="audio/*"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
        />
        <label className="slider-label">
          Threshold ({threshold.toFixed(2)})
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>
        <button type="submit" disabled={!file || loading}>
          {loading ? "Analyzing..." : "Upload and Detect"}
        </button>
        {error ? <p className="error">{error}</p> : null}
      </form>
    </section>
  );
}
