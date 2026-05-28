import json
from pathlib import Path
from typing import Any


class GuidanceEngine:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        kb_dir = self.base_dir / "knowledge_base"

        with (kb_dir / "phishing_types.json").open("r", encoding="utf-8") as fp:
            self.phishing_types: dict[str, Any] = json.load(fp)
        with (kb_dir / "emergency_contacts.json").open("r", encoding="utf-8") as fp:
            self.contacts: dict[str, Any] = json.load(fp)

    def _match_type(self, text: str) -> tuple[str | None, dict[str, Any] | None]:
        lowered = (text or "").lower()
        best_key = None
        best_payload = None
        best_score = 0

        for key, payload in self.phishing_types.items():
            keywords = payload.get("keywords", [])
            score = sum(1 for kw in keywords if kw.lower() in lowered)
            if score > best_score:
                best_key = key
                best_payload = payload
                best_score = score

        return best_key, best_payload

    @staticmethod
    def _default_actions(warning_level: str, risk_score: float) -> list[str]:
        actions = [
            "추가 통화/문자 응답을 중단하고 상대 정보를 기록하세요.",
            "공식 대표번호 또는 가까운 지인에게 사실 여부를 교차 확인하세요.",
            "송금 전이라면 거래를 즉시 중단하고 앱 권한/원격제어 앱 설치 여부를 점검하세요.",
        ]
        if warning_level == "WARNING" or risk_score >= 60:
            actions.insert(0, "고위험 상태입니다. 즉시 통화를 종료하고 신고를 진행하세요.")
        return actions

    def build_guidance(self, risk_score: float, warning_level: str, text: str) -> dict[str, Any]:
        type_key, matched = self._match_type(text)
        if matched is None:
            return {
                "matched_type": "unknown",
                "matched_label": "미분류",
                "summary": "명확한 피싱 유형 키워드가 감지되지 않았습니다.",
                "actions": self._default_actions(warning_level, risk_score),
                "emergency_contacts": self.contacts.get("default", []),
                "banks_notice": self.contacts.get("banks_notice", ""),
            }

        return {
            "matched_type": type_key,
            "matched_label": matched.get("label", type_key),
            "summary": matched.get("summary", ""),
            "actions": matched.get("actions", []),
            "emergency_contacts": self.contacts.get("default", []),
            "banks_notice": self.contacts.get("banks_notice", ""),
        }

