import { useMemo, useRef, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
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
    setScreen(normalized.is_phishing || normalized.max_risk_score >= 60 ? "alert" : "safe");
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
      setError("모델 서버에 연결하지 못했습니다. 샘플 결과로 시연을 계속할 수 있습니다.");
      setScreen("analyzing");
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
          onSamplePhishing={() => useSample("phishing")}
          onSampleSafe={() => useSample("safe")}
        />
      )}

      {screen === "analyze" && (
        <AnalyzeScreen
          error={error}
          isAnalyzing={isAnalyzing}
          onBack={goHome}
          onPickFile={() => fileInputRef.current?.click()}
          onSamplePhishing={() => useSample("phishing")}
          onSampleSafe={() => useSample("safe")}
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
    </div>
  );
}

const iconGlyphs = {
  analytics: "▥",
  arrow_back: "←",
  block: "⊘",
  call: "☎",
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
  upload: "↑",
  verified_user: "✓",
  warning: "!"
};

function Icon({ name, filled = false }) {
  return (
    <span className={`icon ${filled ? "filled" : ""}`} aria-hidden="true">
      {iconGlyphs[name] || name}
    </span>
  );
}

function HomeScreen({ history, onAnalyze, onSamplePhishing, onSampleSafe }) {
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

        <section className="quick-samples">
          <button onClick={onSamplePhishing}>샘플 피싱 통화</button>
          <button onClick={onSampleSafe}>샘플 안전 통화</button>
        </section>

        <section className="history-section">
          <div className="section-row">
            <h3>최근 기록</h3>
            <button>전체 보기</button>
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
            <button>신고 방법</button>
            <button className="danger-soft">긴급 신고</button>
          </div>
        </section>

        <section className="tip-card">
          <div>
            <p>Daily Tip</p>
            <strong>은행은 절대 전화로 PIN 번호를 요구하지 않습니다. 즉시 신고하세요.</strong>
          </div>
        </section>
      </main>

      <BottomNav />
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

function BottomNav() {
  return (
    <nav className="bottom-nav">
      <a className="active" href="#home">
        <Icon name="home" filled />
        Home
      </a>
      <a href="#history">
        <Icon name="history" />
        History
      </a>
      <a href="#settings">
        <Icon name="settings" />
        Settings
      </a>
    </nav>
  );
}

function AnalyzeScreen({ error, isAnalyzing, onBack, onPickFile, onSamplePhishing, onSampleSafe }) {
  return (
    <div className="phone plain">
      <Header title="AIShield" onBack={onBack} />
      <main className="content">
        <section className="analysis-intro">
          <h2>통화 분석하기</h2>
          <p>음성 파일을 업로드하면 모델이 전사, 위험도 계산, 대응 가이드를 한 번에 생성합니다.</p>
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
          <span>위험도</span>
          <strong>{score}%</strong>
          <p>{isDanger ? "피싱 시도 가능성 매우 높음" : "위험 신호 낮음"}</p>
        </section>

        <section className="predicted-type">
          <h3>예측 유형</h3>
          <p>{getLabel(result)}</p>
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
