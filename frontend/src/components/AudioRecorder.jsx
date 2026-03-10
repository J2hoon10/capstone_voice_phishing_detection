import { useEffect, useRef, useState } from "react";

export default function AudioRecorder({ ws }) {
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
    };
  }, []);

  const start = async () => {
    try {
      setError("");
      ws.connect();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          ws.sendBlob(event.data);
        }
      };
      recorder.start(3000);
      setRecording(true);
    } catch (err) {
      setError(err.message || "Mic access failed");
    }
  };

  const stop = () => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    ws.sendJson({ event: "reset" });
    ws.disconnect();
    setRecording(false);
  };

  return (
    <section className="card">
      <h2>Realtime Stream</h2>
      <div className="row">
        <button onClick={recording ? stop : start}>
          {recording ? "Stop Stream" : "Start Stream"}
        </button>
        <span className={ws.isConnected ? "status ok" : "status"}>
          {ws.isConnected ? "WS Connected" : "WS Disconnected"}
        </span>
      </div>
      {error ? <p className="error">{error}</p> : null}
    </section>
  );
}
