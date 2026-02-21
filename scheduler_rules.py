from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from models import CalendarAgentRequest


@dataclass(slots=True)
class ScheduledEvent:
    event_code: str
    title: str
    start: datetime
    end: datetime
    all_day: bool


def parse_date_like(value: Any) -> date | None:
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
    raise ValueError(f"Unsupported date format: {value!r}")


def build_unique_key(user_id: str, event_code: str, event_date: date) -> str:
    return f"{user_id}:{event_code}:{event_date.strftime('%Y%m%d')}"


def generate_schedule(req: CalendarAgentRequest) -> list[ScheduledEvent]:
    tz = ZoneInfo(req.timezone)
    contract_date = parse_date_like(req.contract_date)
    balance_date = parse_date_like(req.balance_date)

    events: list[ScheduledEvent] = []

    def add_all_day(event_code: str, title: str, day: date) -> None:
        start = datetime.combine(day, time.min, tzinfo=tz)
        end = start + timedelta(days=1)
        events.append(ScheduledEvent(event_code=event_code, title=title, start=start, end=end, all_day=True))

    def add_timed(event_code: str, title: str, day: date, hour: int, minute: int, duration_minutes: int) -> None:
        start = datetime.combine(day, time(hour=hour, minute=minute), tzinfo=tz)
        end = start + timedelta(minutes=duration_minutes)
        events.append(ScheduledEvent(event_code=event_code, title=title, start=start, end=end, all_day=False))

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
