from __future__ import annotations

import re

from typing import Optional

from models import AgentContext, AgentRequest, AgentResponse
from sub_agent_base import SubAgentBase


class SubBudgetAgent(SubAgentBase):
    name = "sub_budget_agent"
    description = "예산/대출 상환 가능 범위 추정"
    capabilities = set()

    def run(self, req: AgentRequest, ctx: AgentContext) -> AgentResponse:
        trace = ["sub_budget_agent.run invoked"]
        text = req.text
        monthly_income = _extract_number(text, ["월급", "월소득", "소득", "income"]) or 0
        savings = _extract_number(text, ["보유", "현금", "저축", "자금", "savings"]) or 0

        if monthly_income <= 0 and savings <= 0:
            return AgentResponse(
                success=False,
                message="예산 계산을 위해 월소득 또는 보유자금을 알려주세요. 예: 월소득 500, 보유자금 20000(만원)",
                data={},
                trace=trace,
                errors=["insufficient_input"],
            )

        # 간단한 보수적 추정: 연소득의 4배 + 보유자금
        annual_income = monthly_income * 12
        loan_capacity = annual_income * 4
        total_budget = loan_capacity + savings

        msg = (
            "예상 매수 예산(대략)\n"
            f"- 보유자금: {savings:,.0f}만원\n"
            f"- 추정 대출가능액: {loan_capacity:,.0f}만원\n"
            f"- 총 예산: {total_budget:,.0f}만원\n"
            "주의: 실제 한도는 DSR/LTV, 금리, 지역규제에 따라 달라집니다."
        )
        data = {
            "monthly_income_10k_krw": monthly_income,
            "savings_10k_krw": savings,
            "estimated_loan_capacity_10k_krw": loan_capacity,
            "estimated_total_budget_10k_krw": total_budget,
        }
        return AgentResponse(success=True, message=msg, data=data, trace=trace)


def _extract_number(text: str, hints: list[str]) -> Optional[float]:
    lowered = text.lower()
    if not any(h in lowered for h in hints):
        return None
    numbers = re.findall(r"\d+(?:\.\d+)?", lowered.replace(",", ""))
    if not numbers:
        return None
    return float(numbers[-1])
