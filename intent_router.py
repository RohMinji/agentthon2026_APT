from __future__ import annotations


def classify_intent(text: str) -> str:
    t = (text or "").lower()

    if any(k in t for k in ["전고점", "최고가", "최고점", "현재가 비교"]):
        return "peak_compare"
    if any(k in t for k in ["예산", "대출한도", "월 상환", "dsr", "budget"]):
        return "budget_estimate"
    if any(k in t for k in ["계약일", "잔금일", "매매 일정", "입주", "취득세", "등기", "메일로"]):
        return "buying_plan"
    if any(k in t for k in ["정책", "faq", "pdf", "근거", "출처", "법", "규정"]):
        return "faq_rag"
    if _is_explicit_qa_request(t):
        return "qa_report"
    return "apartment_recommend"


def _is_explicit_qa_request(t: str) -> bool:
    # QA는 명시 요청일 때만 라우팅: 일반 "리포트/평가/로그" 단어는 제외
    keys = [
        "qa 리포트",
        "qa리포트",
        "qa 평가",
        "qa평가",
        "qa 점검",
        "qa점검",
        "quality report",
    ]
    if any(k in t for k in keys):
        return True
    # 영문 qa 단독 토큰
    tokens = t.replace("/", " ").replace("-", " ").split()
    return "qa" in tokens
