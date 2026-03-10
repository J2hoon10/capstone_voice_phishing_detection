import { useMemo, useState } from "react";
import AudioRecorder from "./components/AudioRecorder";
import FileUpload from "./components/FileUpload";
import GuidancePanel from "./components/GuidancePanel";
import RiskGauge from "./components/RiskGauge";
import TranscriptView from "./components/TranscriptView";
import WarningBanner from "./components/WarningBanner";
import useWebSocket from "./hooks/useWebSocket";

function resolveGuidance(payload) {
  if (!payload) {
    return null;
  }
  if (payload.guidance) {
    return payload.guidance;
  }
  return payload;
}

export default function App() {
  const [riskScore, setRiskScore] = useState(0);
  const [warningLevel, setWarningLevel] = useState("NORMAL");
  const [guidance, setGuidance] = useState(null);
  const [transcripts, setTranscripts] = useState([]);

  const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
  const wsUrl = useMemo(() => `${wsProtocol}://${window.location.host}/ws/stream`, [wsProtocol]);

  const ws = useWebSocket(wsUrl, {
    onJsonMessage: (payload) => {
      if (payload?.event !== "prediction" || payload?.status !== "success") {
        return;
      }
      setRiskScore(Number(payload.score || 0));
      setWarningLevel(payload.warning_level || "NORMAL");
      setGuidance(payload.guidance || null);
      if (payload.transcript) {
        setTranscripts((prev) => [payload.transcript, ...prev].slice(0, 20));
      }
    }
  });

  const onFileResult = (result) => {
    setRiskScore(Number(result.max_risk_score || 0));
    setWarningLevel(result.warning_level || "NORMAL");
    setGuidance(resolveGuidance(result.guidance));
    if (result.dangerous_segment) {
      setTranscripts((prev) => [result.dangerous_segment, ...prev].slice(0, 20));
    }
  };

  return (
    <main className="app">
      <header className="hero">
        <p className="eyebrow">Voice Safety Monitor</p>
        <h1>Realtime Voice Phishing Detection</h1>
      </header>

      <section className="grid top">
        <WarningBanner level={warningLevel} />
        <RiskGauge score={riskScore} />
      </section>

      <section className="grid middle">
        <FileUpload onResult={onFileResult} />
        <AudioRecorder ws={ws} />
      </section>

      <section className="grid bottom">
        <GuidancePanel guidance={guidance} />
        <TranscriptView transcripts={transcripts} />
      </section>
    </main>
  );
}
