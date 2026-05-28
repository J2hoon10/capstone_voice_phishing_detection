import { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_API_BASE_URL = "";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(/\/$/, "");

function realtimeWsUrl() {
  if (import.meta.env.VITE_REALTIME_WS_URL) {
    return import.meta.env.VITE_REALTIME_WS_URL;
  }
  const base = new URL(API_BASE_URL || window.location.origin, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/ws/realtime`;
  base.search = "";
  return base.toString();
}
const HISTORY_KEY = "aishield-demo-history";

const samplePhishing = {
  status: "success",
  is_phishing: true,
  max_risk_score: 92,
  warning_level: "WARNING",
  dangerous_segment: "고객님의 예금 계좌에서 무단 인출이 발생했습니다. 계좌 보호를 위해 계좌 번호와 인증번호를 알려주십시오.",
  predicted_label: "수사기관 사칭형",
  guidance: {
    matched_label: "수사기관 사칭형",
    summary:
      "AIShield가 개인정보 요구, 계좌 동결 위협, 긴급한 인증 유도처럼 보이스피싱에서 자주 나타나는 압박 패턴을 감지했습니다.",
    actions: [
      "즉시 통화를 종료하세요.",
      "문자나 전화로 받은 링크를 열지 마세요.",
      "112 또는 금융감독원 1332에 신고하세요."
    ]
  },
  raw: {
    confidence: 0.92,
    class_label: "수사기관 사칭형"
  },
  class_probs: {
    "상품 가입 및 해지": 0.01,
    "이체 출금 대출서비스": 0.02,
    "잔고 및 거래내역": 0.01,
    "수사기관 사칭형": 0.92,
    "대출 사기형": 0.04
  }
};

const sampleSafe = {
  status: "success",
  is_phishing: false,
  max_risk_score: 11,
  warning_level: "NORMAL",
  dangerous_segment: "상담원이 상품 해지 절차와 본인 확인 안내를 일반적인 범위에서 설명했습니다.",
  predicted_label: "상품 가입 및 해지",
  guidance: {
    matched_label: "상품 가입 및 해지",
    summary: "민감한 금융 정보 요구나 긴급 이체 압박이 발견되지 않았습니다.",
    actions: ["통화 내용에 이상이 느껴지면 공식 고객센터 번호로 다시 확인하세요."]
  },
  raw: {
    confidence: 0.89,
    class_label: "상품 가입 및 해지"
  },
  class_probs: {
    "상품 가입 및 해지": 0.89,
    "이체 출금 대출서비스": 0.05,
    "잔고 및 거래내역": 0.03,
    "수사기관 사칭형": 0.01,
    "대출 사기형": 0.02
  }
};

function clampScore(score) {
  const value = Number(score || 0);
  return Math.max(0, Math.min(100, Math.round(value)));
}

function getLabel(result) {
  return (
    result?.predicted_label ||
    result?.pred_label ||
    result?.guidance?.matched_label ||
    result?.raw?.pred_label ||
    result?.raw?.class_label ||
    result?.raw?.predicted_label ||
    (result?.is_phishing ? "보이스피싱 의심" : "정상 통화")
  );
}

function getSummary(result) {
  return (
    result?.guidance?.summary ||
    result?.dangerous_segment ||
    "분석 결과 요약을 준비하지 못했습니다."
  );
}

function getActions(result) {
  const actions = result?.guidance?.actions;
  if (Array.isArray(actions) && actions.length > 0) {
    return actions;
  }
  if (result?.is_phishing) {
    return ["즉시 통화를 종료하세요.", "공식 기관 번호로 다시 확인하세요.", "의심 번호를 신고하고 차단하세요."];
  }
  return ["민감한 정보 요청이 나오면 통화를 중단하고 공식 번호로 확인하세요."];
}

function getClassProbs(result) {
  return result?.class_probs || result?.raw?.class_probs || result?.raw?.probs || null;
}

function buildPhraseCards(result) {
  const text = result?.dangerous_segment || "";
  if (!text.trim()) {
    return [
      {
        tone: "warning",
        phrase: "탐지된 위험 문구가 없습니다.",
        label: "참고 - 추가 확인 필요"
      }
    ];
  }

  const parts = text
    .split(/(?<=[.?!。])\s+/)
    .map((part) => part.trim())
    .filter(Boolean);

  return parts.slice(0, 2).map((phrase, index) => ({
    tone: index === 0 && result?.is_phishing ? "danger" : "warning",
    phrase,
    label:
      index === 0 && result?.is_phishing
        ? "고위험 - 개인정보 또는 압박 표현"
        : "중간 위험 - 추가 확인 필요"
  }));
}

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistoryItem(result, source) {
  const item = {
    id: crypto.randomUUID?.() || String(Date.now()),
    createdAt: new Date().toISOString(),
    source: source || result?.filename || "통화 분석",
    is_phishing: Boolean(result?.is_phishing),
    risk: clampScore(result?.max_risk_score),
    label: getLabel(result)
  };
  const next = [item, ...readHistory()].slice(0, 8);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  return next;
}

async function detectAudio(file) {
  const form = new FormData();
  form.append("audio", file);
  form.append("threshold", "0.5");

  const response = await fetch(`${API_BASE_URL}/api/detect`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }

  return response.json();
}

async function waitForSocketOpen(socket) {
  if (socket.readyState === WebSocket.OPEN) {
    return;
  }
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = () => reject(new Error("실시간 분석 서버에 연결하지 못했습니다."));
  });
}

function downsampleFloat32(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) {
    return buffer;
  }
  const ratio = inputRate / outputRate;
  const newLength = Math.max(1, Math.round(buffer.length / ratio));
  const result = new Float32Array(newLength);
  let offset = 0;
  for (let i = 0; i < newLength; i += 1) {
    const nextOffset = Math.round((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < buffer.length; j += 1) {
      sum += buffer[j];
      count += 1;
    }
    result[i] = count > 0 ? sum / count : 0;
    offset = nextOffset;
  }
  return result;
}

function floatToPcm16(buffer) {
  const pcm = new Int16Array(buffer.length);
  for (let i = 0; i < buffer.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, buffer[i]));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm;
}

function normalizeRealtimePrediction(payload) {
  const risk = Number(payload.risk_score ?? payload.score ?? 0);
  const label = payload.pred_label || payload.raw?.pred_label || (risk >= 60 ? "보이스피싱 의심" : "정상 통화");
  const transcript = payload.dangerous_segment || payload.transcript || "";
  const classProbs = payload.class_probs || payload.raw?.class_probs || {};
  const maxPhishing = Math.max(
    Number(classProbs["대출 사기형"] ?? 0),
    Number(classProbs["수사기관 사칭형"] ?? 0),
  );
  const derivedLevel = maxPhishing > 0.9 ? "WARNING" : maxPhishing > 0.7 ? "CAUTION" : "NORMAL";
  return {
    status: "success",
    is_phishing: Boolean(payload.is_phishing || risk >= 60),
    max_risk_score: risk,
    warning_level: payload.warning_level || derivedLevel,
    dangerous_segment: transcript,
    pred_label: label,
    confidence: payload.confidence,
    class_probs: payload.class_probs,
    guidance: {
      matched_label: label,
      summary: transcript
        ? `실시간 STT 누적 스크립트에서 ${label} 패턴을 감지했습니다.`
        : `실시간 분석에서 ${label} 패턴을 감지했습니다.`,
      actions: risk >= 60
        ? ["즉시 통화를 종료하세요.", "계좌번호나 인증번호를 전달하지 마세요.", "112 또는 금융감독원 1332에 신고하세요."]
        : ["통화 중 민감한 정보 요구가 나오면 공식 번호로 다시 확인하세요."]
    },
    raw: payload
  };
}

export default function App() {
  const [screen, setScreen] = useState("home");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(() => readHistory());
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const score = clampScore(result?.max_risk_score);
  const phrases = useMemo(() => buildPhraseCards(result), [result]);

  const finishAnalysis = (payload, source) => {
    const normalized = {
      ...payload,
      max_risk_score: clampScore(payload?.max_risk_score)
    };
    setResult(normalized);
    setHistory(saveHistoryItem(normalized, source));
    setError("");
    setScreen(normalized.warning_level === "WARNING" ? "alert" : "safe");
  };

  const onFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    setIsAnalyzing(true);
    setError("");
    setScreen("analyzing");
    try {
      const payload = await detectAudio(file);
      finishAnalysis({ ...payload, filename: file.name }, file.name);
    } catch (exc) {
      const fallback = {
        ...samplePhishing,
        filename: file.name,
        guidance: {
          ...samplePhishing.guidance,
          summary:
            "모델 서버에 연결하지 못해 발표용 샘플 결과로 전환했습니다. 실제 배포 환경에서는 업로드된 음성 분석 결과가 표시됩니다."
        }
      };
      finishAnalysis(fallback, `${file.name} · 샘플 fallback`);
      console.error(exc);
    } finally {
      setIsAnalyzing(false);
      event.target.value = "";
    }
  };

  const useSample = (kind) => {
    finishAnalysis(kind === "safe" ? sampleSafe : samplePhishing, kind === "safe" ? "샘플 안전 통화" : "샘플 피싱 통화");
  };

  const goHome = () => setScreen("home");
  const openHistory = () => setScreen("history");
  const openSettings = () => setScreen("settings");
  const openReportGuide = () => setScreen("reportGuide");
  const openEmergency = () => setScreen("emergency");

  return (
    <div className="page-shell">
      <input
        ref={fileInputRef}
        className="sr-only"
        type="file"
        accept="audio/*,.wav,.mp3,.m4a,.webm"
        onChange={onFileChange}
      />

      {screen === "home" && (
        <HomeScreen
          history={history}
          onAnalyze={() => setScreen("analyze")}
          onHistory={openHistory}
          onSettings={openSettings}
          onReportGuide={openReportGuide}
          onEmergency={openEmergency}
        />
      )}

      {screen === "analyze" && (
        <AnalyzeScreen
          error={error}
          isAnalyzing={isAnalyzing}
          onBack={goHome}
          onRealtime={() => setScreen("realtime")}
          onPickFile={() => fileInputRef.current?.click()}
          onSamplePhishing={() => useSample("phishing")}
          onSampleSafe={() => useSample("safe")}
        />
      )}

      {screen === "realtime" && (
        <RealtimeScreen
          onBack={() => setScreen("analyze")}
          onDetected={(payload) => finishAnalysis(payload, "실시간 음성 분석")}
        />
      )}

      {screen === "analyzing" && (
        <AnalyzingScreen
          error={error}
          isAnalyzing={isAnalyzing}
          onBack={() => setScreen("analyze")}
          onPickFile={() => fileInputRef.current?.click()}
          onSamplePhishing={() => useSample("phishing")}
          onSampleSafe={() => useSample("safe")}
        />
      )}

      {screen === "alert" && result && (
        <AlertScreen
          result={result}
          score={score}
          onBack={goHome}
          onDetails={() => setScreen("details")}
          onReport={() => setScreen("reported")}
        />
      )}

      {screen === "safe" && result && (
        <SafeScreen result={result} score={score} onBack={goHome} onDetails={() => setScreen("details")} />
      )}

      {screen === "details" && result && (
        <DetailsScreen
          result={result}
          score={score}
          phrases={phrases}
          onBack={() => setScreen(result.is_phishing ? "alert" : "safe")}
          onDone={goHome}
          onReport={() => setScreen("reported")}
        />
      )}

      {screen === "reported" && result && <ReportedScreen result={result} onDone={goHome} />}

      {screen === "history" && (
        <HistoryScreen
          history={history.length ? history : defaultHistory}
          onBack={goHome}
          onHome={goHome}
          onSettings={openSettings}
        />
      )}

      {screen === "settings" && <SettingsScreen onBack={goHome} onHome={goHome} onHistory={openHistory} />}

      {screen === "reportGuide" && <ReportGuideScreen onBack={goHome} onEmergency={openEmergency} />}

      {screen === "emergency" && (
        <EmergencyScreen
          onBack={goHome}
          onReported={() => {
            setResult(samplePhishing);
            setScreen("reported");
          }}
        />
      )}
    </div>
  );
}

const iconSvg = {
  analytics: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 19V5" />
      <path d="M4 19h16" />
      <path d="M8 16v-5" />
      <path d="M12 16V8" />
      <path d="M16 16v-3" />
      <path d="M20 16V6" />
    </svg>
  ),
  call: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.4 19.4 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 2 .7 2.9a2 2 0 0 1-.5 2.1L8.1 9.9a16 16 0 0 0 6 6l1.2-1.2a2 2 0 0 1 2.1-.5c.9.3 1.9.6 2.9.7A2 2 0 0 1 22 16.9Z" />
    </svg>
  ),
  upload: (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M20 16v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3" />
    </svg>
  )
};

const iconGlyphs = {
  arrow_back: "←",
  block: "⊘",
  check: "✓",
  close: "×",
  done_all: "✓",
  fact_check: "✓",
  history: "↺",
  home: "⌂",
  info: "i",
  notifications: "●",
  search: "⌕",
  settings: "⚙",
  shield: "⬟",
  shield_heart: "♥",
  sync_problem: "!",
  verified_user: "✓",
  warning: "!"
};

function Icon({ name, filled = false }) {
  return (
    <span className={`icon ${filled ? "filled" : ""}`} aria-hidden="true">
      {iconSvg[name] || iconGlyphs[name] || name}
    </span>
  );
}

function HomeScreen({
  history,
  onAnalyze,
  onHistory,
  onSettings,
  onReportGuide,
  onEmergency
}) {
  return (
    <div className="phone">
      <header className="home-header">
        <div className="brand">
          <div className="brand-mark">
            <Icon name="shield" filled />
          </div>
          <h1>AIShield</h1>
        </div>
        <div className="header-actions">
          <button aria-label="검색">
            <Icon name="search" />
          </button>
          <button aria-label="알림">
            <Icon name="notifications" />
          </button>
        </div>
      </header>

      <main className="home-main">
        <section className="greeting">
          <h2>
            안녕하세요, 나현님
            <br />
            기기가 안전하게
            <br />
            보호되고 있습니다.
          </h2>
        </section>

        <section className="status-card">
          <div className="status-orb success">
            <Icon name="check" filled />
          </div>
          <p className="status-title">현재 상태: 안전</p>
          <p className="muted">실시간 보호가 활성화되어 있습니다.</p>
        </section>

        <button className="primary-action" onClick={onAnalyze}>
          <Icon name="analytics" filled />
          통화 분석하기
        </button>


        <section className="history-section">
          <div className="section-row">
            <h3>최근 기록</h3>
            <button onClick={onHistory}>전체 보기</button>
          </div>
          <div className="history-list">
            {(history.length ? history : defaultHistory).slice(0, 2).map((item) => (
              <HistoryItem item={item} key={item.id} />
            ))}
          </div>
        </section>

        <section className="help-card">
          <h4>방금 통화가 의심스러우신가요?</h4>
          <p>보이스피싱을 즉시 식별하고 대처하는 방법을 알아보세요.</p>
          <div className="help-actions">
            <button onClick={onReportGuide}>신고 방법</button>
            <button className="danger-soft" onClick={onEmergency}>
              긴급 신고
            </button>
          </div>
        </section>

        <section className="tip-card">
          <div>
            <p>Daily Tip</p>
            <strong>은행은 절대 전화로 PIN 번호를 요구하지 않습니다. 즉시 신고하세요.</strong>
          </div>
        </section>
      </main>

      <BottomNav active="home" onHome={() => {}} onHistory={onHistory} onSettings={onSettings} />
    </div>
  );
}

const defaultHistory = [
  {
    id: "default-1",
    source: "수신 전화",
    createdAt: new Date(Date.now() - 7200000).toISOString(),
    is_phishing: false,
    risk: 8,
    label: "상품 가입 및 해지"
  },
  {
    id: "default-2",
    source: "알 수 없는 사용자",
    createdAt: new Date(Date.now() - 86400000).toISOString(),
    is_phishing: false,
    risk: 13,
    label: "잔고 및 거래내역"
  }
];

function HistoryItem({ item }) {
  const status = item.is_phishing || item.risk >= 60 ? "위험" : item.risk >= 30 ? "주의" : "안전";
  return (
    <article className="history-item">
      <div className="history-left">
        <div className="call-icon">
          <Icon name="call" />
        </div>
        <div>
          <p className="history-title">{item.source || "수신 전화"}</p>
          <p className="history-meta">
            {relativeTime(item.createdAt)} · {item.label}
          </p>
        </div>
      </div>
      <div className={`history-status ${status === "위험" ? "danger" : status === "주의" ? "warning" : "safe"}`}>
        <span />
        {status}
      </div>
    </article>
  );
}

function relativeTime(value) {
  const diff = Date.now() - new Date(value).getTime();
  if (diff < 60000) return "방금 전";
  if (diff < 3600000) return `${Math.round(diff / 60000)}분 전`;
  if (diff < 86400000) return `${Math.round(diff / 3600000)}시간 전`;
  return "어제";
}

function BottomNav({ active = "home", onHome, onHistory, onSettings }) {
  return (
    <nav className="bottom-nav">
      <button className={active === "home" ? "active" : ""} onClick={onHome}>
        <Icon name="home" filled />
        Home
      </button>
      <button className={active === "history" ? "active" : ""} onClick={onHistory}>
        <Icon name="history" />
        History
      </button>
      <button className={active === "settings" ? "active" : ""} onClick={onSettings}>
        <Icon name="settings" />
        Settings
      </button>
    </nav>
  );
}

function HistoryScreen({ history, onBack, onHome, onSettings }) {
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content history-page">
        <section className="details-heading">
          <h2>최근 기록</h2>
          <p>최근 분석된 통화와 위험도를 확인하세요.</p>
        </section>
        <div className="history-list full">
          {history.map((item) => (
            <HistoryItem item={item} key={item.id} />
          ))}
        </div>
      </main>
      <BottomNav active="history" onHome={onHome} onHistory={() => {}} onSettings={onSettings} />
    </div>
  );
}

function SettingsScreen({ onBack, onHome, onHistory }) {
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content settings-page">
        <section className="details-heading">
          <h2>Settings</h2>
          <p>발표 데모용 보호 설정 상태입니다.</p>
        </section>
        <section className="settings-panel">
          <SettingRow title="실시간 보호" description="통화 중 위험 표현을 감지합니다." enabled />
          <SettingRow title="긴급 알림" description="고위험 통화 탐지 시 즉시 경고합니다." enabled />
          <SettingRow title="최근 기록 저장" description="분석 결과를 기기 내 기록으로 남깁니다." enabled />
        </section>
      </main>
      <BottomNav active="settings" onHome={onHome} onHistory={onHistory} onSettings={() => {}} />
    </div>
  );
}

function SettingRow({ title, description, enabled }) {
  return (
    <article className="setting-row">
      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
      <span className={`toggle ${enabled ? "on" : ""}`} aria-hidden="true">
        <span />
      </span>
    </article>
  );
}

function ReportGuideScreen({ onBack, onEmergency }) {
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content guide-page">
        <section className="details-heading">
          <h2>신고 방법</h2>
          <p>보이스피싱이 의심될 때 바로 따라야 할 대응 순서입니다.</p>
        </section>
        <section className="guide-steps">
          <GuideStep title="통화 종료" description="상대가 기관명이나 은행명을 말해도 우선 전화를 끊습니다." />
          <GuideStep title="공식 번호 확인" description="112, 금융감독원 1332, 은행 공식 고객센터로 직접 확인합니다." />
          <GuideStep title="계좌 보호" description="계좌 이체, 원격 제어 앱 설치, 인증번호 전달은 즉시 중단합니다." />
        </section>
        <button className="primary-action compact danger-action" onClick={onEmergency}>
          긴급 신고 화면 열기
        </button>
      </main>
    </div>
  );
}

function GuideStep({ title, description }) {
  return (
    <article className="guide-step">
      <Icon name="verified_user" />
      <div>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </article>
  );
}

function EmergencyScreen({ onBack, onReported }) {
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content emergency-page">
        <section className="emergency-card">
          <div className="alert-icon">
            <Icon name="warning" filled />
          </div>
          <h2>긴급 신고</h2>
          <p>개인정보나 인증번호를 요구받았다면 통화를 종료하고 공식 신고 채널로 확인하세요.</p>
          <div className="emergency-lines">
            <strong>112</strong>
            <span>경찰청 긴급 신고</span>
          </div>
          <div className="emergency-lines">
            <strong>1332</strong>
            <span>금융감독원 상담</span>
          </div>
        </section>
        <button className="primary-action compact danger-action" onClick={onReported}>
          신고 완료 처리
        </button>
      </main>
    </div>
  );
}

function RealtimeScreen({ onBack, onDetected }) {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState("대기 중");
  const [error, setError] = useState("");
  const [duration, setDuration] = useState(0);
  const [transcripts, setTranscripts] = useState([]);
  const [latest, setLatest] = useState(null);
  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const analyserRef = useRef(null);
  const processorRef = useRef(null);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const triggeredRef = useRef(false);

  const stopWaveform = () => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * pixelRatio));
    const height = Math.max(1, Math.floor(rect.height * pixelRatio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    const data = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(data);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8fbff";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(0, 100, 255, 0.13)";
    ctx.lineWidth = 1 * pixelRatio;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    ctx.strokeStyle = "#0064ff";
    ctx.lineWidth = 2.5 * pixelRatio;
    ctx.beginPath();
    const sliceWidth = width / data.length;
    for (let index = 0; index < data.length; index += 1) {
      const x = index * sliceWidth;
      const y = (data[index] / 255) * height;
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    animationRef.current = requestAnimationFrame(drawWaveform);
  };

  const cleanup = () => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ event: "stop" }));
      socket.close();
    }
    stopWaveform();
    processorRef.current?.disconnect();
    analyserRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    audioContextRef.current?.close?.();
    socketRef.current = null;
    streamRef.current = null;
    audioContextRef.current = null;
    sourceRef.current = null;
    analyserRef.current = null;
    processorRef.current = null;
    setIsRunning(false);
  };

  useEffect(() => () => cleanup(), []);

  const start = async () => {
    setError("");
    setTranscripts([]);
    setLatest(null);
    setDuration(0);
    triggeredRef.current = false;
    try {
      setStatus("마이크 권한 요청 중");
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      const socket = new WebSocket(realtimeWsUrl());
      socket.binaryType = "arraybuffer";
      await waitForSocketOpen(socket);

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioContext = new AudioContextClass();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.72;

      socket.onmessage = (event) => {
        let payload;
        try {
          payload = JSON.parse(event.data);
        } catch {
          return;
        }
        if (payload.event === "ready") {
          setStatus("실시간 분석 중");
        } else if (payload.event === "audio") {
          setDuration(Number(payload.duration || 0));
        } else if (payload.event === "transcript") {
          setTranscripts((prev) => [payload, ...prev].slice(0, 6));
        } else if (payload.event === "prediction" && payload.status === "success") {
          setLatest(payload);
          const normalized = normalizeRealtimePrediction(payload);
          if (!triggeredRef.current && normalized.warning_level === "WARNING") {
            triggeredRef.current = true;
            setStatus("위험 통화 감지");
            cleanup();
            setTimeout(() => onDetected(normalized), 250);
          }
        } else if (payload.event === "error") {
          setError(payload.detail || "실시간 분석 중 오류가 발생했습니다.");
          setStatus("오류");
        }
      };
      socket.onclose = () => {
        if (!triggeredRef.current) {
          setStatus("연결 종료");
          setIsRunning(false);
        }
      };

      processor.onaudioprocess = (event) => {
        if (socket.readyState !== WebSocket.OPEN) {
          return;
        }
        const input = event.inputBuffer.getChannelData(0);
        const downsampled = downsampleFloat32(input, audioContext.sampleRate, 16000);
        const pcm = floatToPcm16(downsampled);
        socket.send(pcm.buffer);
      };

      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioContext.destination);

      socketRef.current = socket;
      streamRef.current = stream;
      audioContextRef.current = audioContext;
      sourceRef.current = source;
      analyserRef.current = analyser;
      processorRef.current = processor;
      animationRef.current = requestAnimationFrame(drawWaveform);
      setIsRunning(true);
      setStatus("실시간 분석 중");
    } catch (exc) {
      cleanup();
      setError(exc.message || "마이크 또는 실시간 분석 서버를 사용할 수 없습니다.");
      setStatus("오류");
    }
  };

  const stop = () => {
    if (latest) {
      const normalized = normalizeRealtimePrediction(latest);
      cleanup();
      setTimeout(() => onDetected(normalized), 250);
    } else {
      cleanup();
      setStatus("중지됨");
    }
  };

  const risk = clampScore(latest?.risk_score ?? latest?.score ?? 0);
  const warningLevel = latest?.warning_level || "NORMAL";

  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content realtime-page">
        <section className={`live-status ${warningLevel.toLowerCase()}`}>
          <div>
            <span>Live Monitor</span>
            <h2>{status}</h2>
            <p>{duration.toFixed(1)}초 수신 · 20초 STT 창 / 10초 overlap · 0.5초 모델 폴링</p>
          </div>
          <strong>{risk}%</strong>
        </section>

        <section className="live-controls">
          <button className="primary-action compact" disabled={isRunning} onClick={start}>
            마이크 시작
          </button>
          <button className="stop-action" disabled={!isRunning} onClick={stop}>
            중지
          </button>
        </section>

        {error ? <p className="error-text">{error}</p> : null}

        <section className={`voice-wave-card ${isRunning ? "active" : ""}`}>
          <div>
            <h3>마이크 입력</h3>
            <span>{isRunning ? "수신 중" : "대기 중"}</span>
          </div>
          <canvas ref={canvasRef} aria-label="실시간 마이크 파형" />
        </section>

        <section className="live-card">
          <h3>현재 모델 판단</h3>
          <p>{latest?.pred_label || "스크립트가 쌓이면 판단을 표시합니다."}</p>
          {latest?.class_probs ? <ClassProbabilities result={{ class_probs: latest.class_probs }} /> : null}
        </section>

        <section className="live-card">
          <h3>실시간 STT 스크립트</h3>
          <div className="live-transcripts">
            {transcripts.length ? (
              transcripts.map((item) => (
                <article key={`${item.window_start}-${item.window_end}`}>
                  <span>{Number(item.window_start).toFixed(1)}s - {Number(item.window_end).toFixed(1)}s</span>
                  <p>{item.text}</p>
                </article>
              ))
            ) : (
              <p>마이크를 시작하면 20초 단위 전사 결과가 여기에 누적됩니다.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function AnalyzeScreen({ error, isAnalyzing, onBack, onRealtime, onPickFile, onSamplePhishing, onSampleSafe }) {
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content">
        <section className="analysis-intro">
          <h2>통화 분석하기</h2>
          <p>실시간 마이크 입력 또는 음성 파일로 STT, 위험도 계산, 대응 가이드를 생성합니다.</p>
        </section>

        <section className="upload-card realtime-entry">
          <div className="upload-icon live">
            <Icon name="call" filled />
          </div>
          <h3>실시간 음성 분석</h3>
          <p>마이크 음성을 20초 창과 10초 overlap으로 전사하고, 0.5초마다 탐지 모델을 갱신합니다.</p>
          <button className="primary-action compact" onClick={onRealtime}>
            실시간 분석 시작
          </button>
        </section>

        <section className="upload-card">
          <div className="upload-icon">
            <Icon name="upload" filled />
          </div>
          <h3>음성 파일 선택</h3>
          <p>wav, mp3, m4a, webm 파일을 지원합니다.</p>
          <button className="primary-action compact" disabled={isAnalyzing} onClick={onPickFile}>
            파일 업로드
          </button>
        </section>

        {error && <ErrorBox message={error} onSamplePhishing={onSamplePhishing} onSampleSafe={onSampleSafe} />}

        <section className="sample-panel">
          <h3>발표용 빠른 시연</h3>
          <div>
            <button onClick={onSamplePhishing}>샘플 피싱 결과 보기</button>
            <button onClick={onSampleSafe}>샘플 안전 결과 보기</button>
          </div>
        </section>
      </main>
    </div>
  );
}

function AnalyzingScreen({ error, isAnalyzing, onBack, onPickFile, onSamplePhishing, onSampleSafe }) {
  return (
    <div className="phone plain centered">
      <Header title="AIShield" onBack={onBack} />
      <main className="content center-content">
        {isAnalyzing ? (
          <>
            <div className="loader-ring" />
            <h2>통화를 분석 중입니다</h2>
            <p>전사 내용을 확인하고 위험도와 의심 문구를 계산하고 있습니다.</p>
            <div className="analysis-steps">
              <span>음성 업로드</span>
              <span>STT 전사</span>
              <span>피싱 분류</span>
            </div>
          </>
        ) : (
          <>
            <div className="upload-icon error">
              <Icon name="sync_problem" filled />
            </div>
            <h2>분석을 완료하지 못했습니다</h2>
            <p>{error || "다시 시도하거나 샘플 결과로 데모를 계속하세요."}</p>
            <button className="primary-action compact" onClick={onPickFile}>
              다시 업로드
            </button>
            <ErrorBox message={error} onSamplePhishing={onSamplePhishing} onSampleSafe={onSampleSafe} />
          </>
        )}
      </main>
    </div>
  );
}

function AlertScreen({ result, score, onBack, onDetails, onReport }) {
  return (
    <div className="alert-shell">
      <button className="floating-back" onClick={onBack} aria-label="뒤로">
        <Icon name="arrow_back" />
      </button>
      <section className="alert-modal">
        <div className="alert-body">
          <div className="alert-icon">
            <Icon name="shield_heart" filled />
            <span>
              <Icon name="close" />
            </span>
          </div>
          <h1>Phishing Detected</h1>
          <p>
            현재 통화는 보이스피싱일 확률이 매우 높습니다.
            <br />
            <strong>절대 개인정보를 제공하지 마시고</strong> 즉시 통화를 종료하세요.
          </p>
          <div className="confidence">
            <Icon name="analytics" />
            탐지 확신도: {score}%
          </div>
          <div className="alert-actions">
            <button className="end-call" onClick={onBack}>
              통화 종료
            </button>
            <button className="outline-danger" onClick={onReport}>
              이 번호 신고하기
            </button>
            <button className="link-button" onClick={onDetails}>
              상세 분석 보기
            </button>
          </div>
        </div>
        <div className="alert-footer">
          <Icon name="info" filled />
          <div>
            <h3>어떻게 탐지했나요?</h3>
            <p>{getSummary(result)}</p>
          </div>
        </div>
      </section>
      <p className="protected">Protected by AIShield</p>
    </div>
  );
}

function SafeScreen({ result, score, onBack, onDetails }) {
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content safe-result">
        <div className="status-orb success large">
          <Icon name="check" filled />
        </div>
        <h2>안전한 통화로 판단됩니다</h2>
        <p>민감한 개인정보 요구나 강한 압박 표현이 발견되지 않았습니다.</p>
        <div className="safe-score">
          <span>위험도</span>
          <strong>{score}%</strong>
        </div>
        <section className="result-card">
          <h3>예측 유형</h3>
          <p>{getLabel(result)}</p>
        </section>
        <button className="primary-action compact" onClick={onDetails}>
          분석 상세 보기
        </button>
      </main>
    </div>
  );
}

function DetailsScreen({ result, score, phrases, onBack, onDone, onReport }) {
  const actions = getActions(result);
  const isDanger = result.is_phishing || score >= 60;
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="details-main">
        <section className="details-heading">
          <h2>피싱 탐지 상세 정보</h2>
          <p>최근 통화 분석 결과</p>
        </section>

        <section className={`risk-card ${isDanger ? "danger" : "safe"}`}>
          <span>탐지 결과</span>
          <strong style={{ fontSize: "28px", lineHeight: "1.3", margin: "8px 0" }}>
            {getLabel(result)}
          </strong>
          <p>신뢰도 {Math.round((result?.confidence ?? 0) * 100)}%</p>
        </section>

        <ClassProbabilities result={result} />

        <section>
          <h3 className="section-title">
            <Icon name={isDanger ? "warning" : "fact_check"} filled />
            {isDanger ? "의심스러운 문구" : "분석된 통화 내용"}
          </h3>
          <div className="phrase-list">
            {phrases.map((item, index) => (
              <article className="phrase-card" key={`${item.phrase}-${index}`}>
                <span className={`phrase-bar ${item.tone}`} />
                <div>
                  <p>"{item.phrase}"</p>
                  <strong className={item.tone}>{item.label}</strong>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="ai-analysis">
          <h3>AI 분석</h3>
          <p>{getSummary(result)}</p>
        </section>

        <section className="tips-section">
          <h3>안전 수칙</h3>
          {actions.slice(0, 3).map((action, index) => (
            <div className="tip-row" key={action}>
              <div>
                <Icon name={index === 0 ? "verified_user" : index === 1 ? "call" : "block"} />
              </div>
              <p>{action}</p>
            </div>
          ))}
        </section>
      </main>
      <footer className="fixed-actions">
        <button onClick={onDone}>완료</button>
        <button className="report" onClick={onReport}>
          이 번호 신고하기
        </button>
      </footer>
    </div>
  );
}

function ClassProbabilities({ result }) {
  const probs = getClassProbs(result);
  if (!probs) {
    return null;
  }

  const entries = Object.entries(probs).map(([label, value]) => [
    label,
    value <= 1 ? Math.round(value * 100) : Math.round(value)
  ]);

  return (
    <section className="class-probs">
      <h3>5-Class 확률</h3>
      {entries.map(([label, value]) => (
        <div className="prob-row" key={label}>
          <div>
            <span>{label}</span>
            <strong>{value}%</strong>
          </div>
          <progress max="100" value={value} />
        </div>
      ))}
    </section>
  );
}

function ReportedScreen({ result, onDone }) {
  return (
    <div className="phone plain centered">
      <main className="content center-content">
        <div className="status-orb success large">
          <Icon name="done_all" filled />
        </div>
        <h2>신고가 접수되었습니다</h2>
        <p>{getLabel(result)} 분석 결과와 위험 문구가 신고 기록에 저장되었습니다.</p>
        <button className="primary-action compact" onClick={onDone}>
          홈으로 돌아가기
        </button>
      </main>
    </div>
  );
}

function Header({ title, onBack }) {
  return (
    <header className="simple-header">
      <button onClick={onBack} aria-label="뒤로">
        <Icon name="arrow_back" />
      </button>
      <h1>{title}</h1>
    </header>
  );
}

function ErrorBox({ message, onSamplePhishing, onSampleSafe }) {
  if (!message) {
    return null;
  }
  return (
    <section className="error-box">
      <strong>데모 fallback</strong>
      <p>{message}</p>
      <div>
        <button onClick={onSamplePhishing}>피싱 샘플</button>
        <button onClick={onSampleSafe}>안전 샘플</button>
      </div>
    </section>
  );
}
