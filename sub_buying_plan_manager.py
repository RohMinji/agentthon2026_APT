from __future__ import annotations

import os
import smtplib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from email.message import EmailMessage
from typing import Any, Optional
from zoneinfo import ZoneInfo

from models import AgentContext, AgentRequest, AgentResponse
from sub_agent_base import SubAgentBase

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass
class PlanEvent:
    event_code: str
    title: str
    start: datetime
    end: datetime
    all_day: bool


@dataclass
class BuyingPlanRequest:
    user_id: str
    timezone: str = "Asia/Seoul"
    contract_date: Optional[date] = None
    balance_date: Optional[date] = None
    loan: bool = False
    include_funding_plan: bool = True
    include_movein: bool = True
    dry_run: bool = False
    email_to: Optional[str] = None


@dataclass
class BuyingPlanResponse:
    success: bool
    events: list[PlanEvent] = field(default_factory=list)
    email_sent: bool = False
    errors: list[str] = field(default_factory=list)
    subject: str = ""
    mail_body: str = ""
    human_summary: str = ""


def parse_date_like(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    raise ValueError(f"지원하지 않는 날짜 형식입니다: {value!r}")


def generate_schedule(req: BuyingPlanRequest) -> list[PlanEvent]:
    tz = ZoneInfo(req.timezone)
    contract_date = parse_date_like(req.contract_date)
    balance_date = parse_date_like(req.balance_date)

    events: list[PlanEvent] = []

    def add_all_day(event_code: str, title: str, day: date) -> None:
        start = datetime.combine(day, time.min, tzinfo=tz)
        end = start + timedelta(days=1)
        events.append(PlanEvent(event_code=event_code, title=title, start=start, end=end, all_day=True))

    def add_timed(event_code: str, title: str, day: date, hour: int, minute: int, duration_minutes: int) -> None:
        start = datetime.combine(day, time(hour=hour, minute=minute), tzinfo=tz)
        end = start + timedelta(minutes=duration_minutes)
        events.append(PlanEvent(event_code=event_code, title=title, start=start, end=end, all_day=False))

    if contract_date:
        add_all_day("txn_report", "부동산 거래신고(실거래 신고) 마감", contract_date + timedelta(days=30))
        if req.include_funding_plan:
            add_all_day("funding_plan", "자금조달계획서 제출 마감", contract_date + timedelta(days=30))
        add_all_day("rights_check_contract", "등기부등본/권리관계 확인(계약 전)", contract_date - timedelta(days=1))

    if balance_date:
        add_all_day("acq_tax", "취득세 신고/납부 마감", balance_date + timedelta(days=60))
        add_all_day("ownership_transfer", "소유권 이전등기 신청 마감", balance_date + timedelta(days=60))

        add_all_day("rights_check_balance", "등기부등본/권리관계 확인(잔금 전)", balance_date - timedelta(days=1))
        add_timed("rights_check_same_day", "등기부등본/권리관계 최종 확인", balance_date, 9, 0, 30)

        if req.loan:
            add_all_day("loan_prep_start", "대출 서류 준비/은행 상담 시작", balance_date - timedelta(days=21))
            add_all_day("loan_final_review", "대출 실행 확정/서류 최종 점검", balance_date - timedelta(days=7))
            add_all_day("loan_transfer_check", "대출 실행/송금 최종 확인", balance_date - timedelta(days=1))

        if req.include_movein:
            add_timed("key_handover", "열쇠 인수/입주 체크리스트", balance_date, 10, 0, 60)
            add_timed("address_change", "전입신고/주소변경 체크", balance_date + timedelta(days=1), 11, 0, 30)

    return sorted(events, key=lambda e: e.start)


def get_policy_snapshot(today: date) -> list[str]:
    # 정책 상세 수치/요건은 변동 가능성이 있으므로, 메일에는 핵심 체크포인트 중심으로 제공
    # 실행 시점 날짜를 함께 표기해 최신성 기준을 명확히 한다.
    return [
        f"정책 체크 기준일: {today.isoformat()}",
        "1) 실거래 신고: 계약 후 30일 이내 신고 필요 여부 확인",
        "2) 취득세: 취득일(통상 잔금일) 기준 60일 내 신고/납부 확인",
        "3) 소유권 이전등기: 잔금 이후 지연 없이 신청(통상 60일 내 준비 권장)",
        "4) 조정대상지역/투기과열지구 여부에 따라 대출·세금 요건이 달라질 수 있으니 계약 전 최신 공고 확인",
        "5) 생애최초/신혼부부/청년 등 정책대출·세제혜택은 시점별 요건이 바뀔 수 있어 사전 재확인 필요",
        "참고: 국토교통부, 홈택스, 정부24, 지자체 고지사항",
    ]


def get_trade_checklist() -> list[str]:
    return [
        "매도인 신분/대리권 및 권리관계(근저당·가압류·가처분) 확인",
        "계약금·중도금·잔금 지급 계좌/증빙 관리(이체내역, 영수증)",
        "특약사항(하자, 수리, 명도일, 위약 조항) 문구 명확화",
        "관리비/공과금/체납 정산 기준일 확인",
        "잔금일 당일 등기 서류, 대출 실행, 열쇠 인수 절차 동선 점검",
        "입주 후 전입신고·확정일자·주소변경/자동이체 변경",
    ]


class GmailSender:
    def __init__(self, smtp_user: Optional[str] = None, smtp_password: Optional[str] = None):
        self.smtp_user = smtp_user or os.getenv("GMAIL_SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("GMAIL_APP_PASSWORD")

    def send(self, to_email: str, subject: str, body: str) -> None:
        if not self.smtp_user or not self.smtp_password:
            raise RuntimeError("GMAIL_SMTP_USER/GMAIL_APP_PASSWORD 환경변수가 필요합니다.")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = to_email
        msg.set_content(body)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)


class BuyingPlanManager:
    """오케스트레이터에서 호출하는 부동산 매매 일정/정책 메일 매니저."""

    def __init__(self, mail_sender: Optional[GmailSender] = None, default_to: Optional[str] = None):
        self.mail_sender = mail_sender or GmailSender()
        self.default_to = default_to or os.getenv("GMAIL_TO")

    def run(self, req: BuyingPlanRequest) -> BuyingPlanResponse:
        errors: list[str] = []

        try:
            req = self._normalize_request(req)
        except ValueError as exc:
            return self._response(False, [], False, [str(exc)], "", "")

        if req.contract_date is None and req.balance_date is None:
            return self._response(
                False,
                [],
                False,
                ["계약일(contract_date) 또는 잔금일(balance_date) 중 최소 하나는 필요합니다."],
                "",
                "",
            )

        events = generate_schedule(req)
        subject, body = self._build_mail_content(req, events)

        if req.dry_run:
            return self._response(True, events, False, [], subject, body)

        to_email = req.email_to or self.default_to
        if not to_email:
            return self._response(False, events, False, ["수신 이메일이 설정되지 않았습니다. 관리자에게 수신 주소 설정을 요청해주세요."], subject, body)

        try:
            self.mail_sender.send(to_email=to_email, subject=subject, body=body)
            return self._response(True, events, True, [], subject, body)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"메일 발송 실패: {exc}")
            return self._response(False, events, False, errors, subject, body)

    def _normalize_request(self, req: BuyingPlanRequest) -> BuyingPlanRequest:
        contract_date = self._safe_date(req.contract_date, "contract_date")
        balance_date = self._safe_date(req.balance_date, "balance_date")
        return BuyingPlanRequest(
            user_id=req.user_id,
            timezone=req.timezone,
            contract_date=contract_date,
            balance_date=balance_date,
            loan=req.loan,
            include_funding_plan=req.include_funding_plan,
            include_movein=req.include_movein,
            dry_run=req.dry_run,
            email_to=req.email_to,
        )

    @staticmethod
    def _safe_date(value: Any, field_name: str) -> Optional[date]:
        try:
            return parse_date_like(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} 값이 올바른 날짜 형식이 아닙니다. 예: 2026-03-01") from exc

    def _build_mail_content(self, req: BuyingPlanRequest, events: list[PlanEvent]) -> tuple[str, str]:
        today = datetime.now(ZoneInfo(req.timezone)).date()
        subject = f"[부동산 매매 플랜] {today.isoformat()} / user={req.user_id}"

        lines = [
            "부동산 매매 일정/정책 안내 메일",
            "",
            f"- 생성시각: {datetime.now(ZoneInfo(req.timezone)).strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"- 사용자: {req.user_id}",
            f"- 계약일: {req.contract_date.isoformat() if req.contract_date else '미입력'}",
            f"- 잔금일: {req.balance_date.isoformat() if req.balance_date else '미입력'}",
            f"- 대출 여부: {'예' if req.loan else '아니오'}",
            "",
            f"[필수/권장 일정] 총 {len(events)}건",
        ]

        for idx, ev in enumerate(events, start=1):
            when = ev.start.date().isoformat() if ev.all_day else ev.start.strftime("%Y-%m-%d %H:%M")
            lines.append(f"{idx}. {when} | {ev.title}")

        lines.append("")
        lines.append("[현재 기준 정책 체크포인트]")
        lines.extend(get_policy_snapshot(today))

        lines.append("")
        lines.append("[아파트 매매 핵심 체크리스트]")
        for item in get_trade_checklist():
            lines.append(f"- {item}")

        lines.append("")
        lines.append("※ 본 메일은 일정 관리 보조용입니다. 최종 법적 판단/신고기한은 관할 기관 공지와 전문가 확인을 우선하세요.")

        return subject, "\n".join(lines)

    def _response(
        self,
        success: bool,
        events: list[PlanEvent],
        email_sent: bool,
        errors: list[str],
        subject: str,
        body: str,
    ) -> BuyingPlanResponse:
        summary_lines = [f"[필수/권장 일정] 총 {len(events)}건"]
        for idx, ev in enumerate(events, start=1):
            when = ev.start.date().isoformat() if ev.all_day else ev.start.strftime("%Y-%m-%d %H:%M")
            summary_lines.append(f"{idx}. {when} | {ev.title}")

        if email_sent:
            summary_lines.append("")
            summary_lines.append("메일 발송을 완료했습니다.")

        if errors:
            summary_lines.append("")
            summary_lines.append("⚠️ 주의/오류")
            for e in errors:
                summary_lines.append(f"- {e}")

        return BuyingPlanResponse(
            success=success,
            events=events,
            email_sent=email_sent,
            errors=errors,
            subject=subject,
            mail_body=body,
            human_summary="\n".join(summary_lines),
        )


__all__ = [
    "BuyingPlanManager",
    "BuyingPlanRequest",
    "BuyingPlanResponse",
    "PlanEvent",
    "GmailSender",
    "generate_schedule",
    "parse_date_like",
]


class SubBuyingPlanManagerAgent(SubAgentBase):
    name = "sub_buying_plan_manager"
    description = "매매 일정/주의사항 정리 및 메일 발송"
    capabilities = {"email_send"}

    def __init__(self):
        self.manager = BuyingPlanManager()

    def run(self, req: AgentRequest, ctx: AgentContext) -> AgentResponse:
        trace = ["sub_buying_plan_manager.run invoked", "extracting dates/options from text"]
        contract_date = _find_date(req.text, ["계약"])
        balance_date = _find_date(req.text, ["잔금"])
        loan = any(k in req.text for k in ["대출", "loan"])

        if contract_date is None and balance_date is None:
            return AgentResponse(
                success=False,
                message="계약일 또는 잔금일을 YYYY-MM-DD 형식으로 포함해주세요.",
                data={},
                trace=trace,
                errors=["missing_dates"],
            )

        dry_run = not req.metadata.get("confirm_send", False)
        if not dry_run and "email_send" not in ctx.capabilities:
            return AgentResponse(
                success=False,
                message="메일 발송 권한이 없어 요청을 차단했습니다.",
                data={},
                trace=trace,
                errors=["capability_blocked:email_send"],
            )

        bp_req = BuyingPlanRequest(
            user_id=req.user_id,
            timezone=req.timezone,
            contract_date=contract_date,
            balance_date=balance_date,
            loan=loan,
            dry_run=dry_run,
        )
        result = self.manager.run(bp_req)
        trace.append(f"dry_run={dry_run}")
        return AgentResponse(
            success=result.success,
            message=result.human_summary,
            data={"subject": result.subject, "email_sent": result.email_sent, "event_count": len(result.events)},
            trace=trace,
            errors=result.errors,
        )


def _find_date(text: str, hint_words: list[str]) -> Optional[date]:
    pattern = re.compile(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
    lowered = text.lower()
    if not any(w in lowered for w in hint_words):
        return None
    m = pattern.search(text)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d)
    except ValueError:
        return None
