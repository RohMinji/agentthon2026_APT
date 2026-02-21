from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field

from models import AgentContext, AgentRequest, AgentResponse
from sub_agent_base import SubAgentBase
from sub_apartment_agent import create_apartment_search_agent

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "si_do": {"type": ["string", "null"]},
        "si_gungu": {"type": ["string", "null"]},
        "eupmyeondong": {"type": ["string", "null"]},
        "corridor_type": {"type": ["string", "null"], "enum": ["계단식", "복도식", None]},
        "heating_type": {"type": ["string", "null"], "enum": ["개별난방", "지역난방", "중앙난방", None]},
        "min_households": {"type": ["integer", "null"]},
        "max_households": {"type": ["integer", "null"]},
        "min_parking_per_household": {"type": ["number", "null"]},
        "max_parking_per_household": {"type": ["number", "null"]},
        "min_age": {"type": ["integer", "null"]},
        "max_age": {"type": ["integer", "null"]},
        "min_exclusive_area": {"type": ["number", "null"]},
        "max_exclusive_area": {"type": ["number", "null"]},
        "min_price_eok": {"type": ["number", "null"]},
        "max_price_eok": {"type": ["number", "null"]},
    },
    "required": [
        "si_do",
        "si_gungu",
        "eupmyeondong",
        "corridor_type",
        "heating_type",
        "min_households",
        "max_households",
        "min_parking_per_household",
        "max_parking_per_household",
        "min_age",
        "max_age",
        "min_exclusive_area",
        "max_exclusive_area",
        "min_price_eok",
        "max_price_eok",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
역할: 아파트 검색 필터 추출기.

출력:
- JSON 객체만 출력.
- 키는 사전에 정의된 필터 키만 사용.
- 값이 없으면 null.

규칙:
1) 면적 변환
- N평 -> N * 3.3058 (㎡)

2) 가격 변환
- N만원 -> N * 0.00001 (억원)
- N억 -> N (억원)

3) 범위 해석
- 이상/초과 -> min_*
- 이하/미만 -> max_*

4) 단일 목표값 보정
- 특정 면적(예: 84㎡, 33평): min_exclusive_area=값-10, max_exclusive_area=값+10
- 특정 가격(예: 15억): min_price_eok=값-1, max_price_eok=값+1

5) 다중 지역
- "송파구 또는 서초구" -> "송파구,서초구"

6) 멀티턴
- 입력에 previous_filters가 있으면 이를 기본값으로 사용.
- user_query에서 언급된 항목만 수정.
- remove_conditions가 있으면 해당 조건 키를 null로 설정.
"""


@dataclass
class _ApartmentSession:
    user_history: deque
    last_filters: dict = field(default_factory=dict)


class SubApartmentRecommendAgent(SubAgentBase):
    name = "sub_apartment_agent"
    description = "조건 기반 아파트 추천"
    capabilities = set()

    def __init__(self, csv_path: str = "./DATA/apt_basic_info.csv"):
        self.csv_path = csv_path
        self.agent = None
        self._openai_client = None
        self._sessions: dict[str, _ApartmentSession] = {}
        self._sessions_lock = threading.Lock()

    def reset_user_session(self, user_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(user_id, None)

    def _get_session(self, user_id: str) -> _ApartmentSession:
        with self._sessions_lock:
            if user_id not in self._sessions:
                self._sessions[user_id] = _ApartmentSession(user_history=deque(maxlen=6))
            return self._sessions[user_id]

    def _get_openai_client(self):
        if OpenAI is None:
            return None
        if self._openai_client is not None:
            return self._openai_client
        api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
        endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        if not (api_key and endpoint):
            return None
        # 기존 동작과 호환되도록 Azure OpenAI v1 base_url 형식 유지
        self._openai_client = OpenAI(base_url=f"{endpoint}/openai/V1/", api_key=api_key)
        return self._openai_client

    def _empty_filter_payload(self) -> dict:
        return {
            "si_do": None,
            "si_gungu": None,
            "eupmyeondong": None,
            "corridor_type": None,
            "heating_type": None,
            "min_households": None,
            "max_households": None,
            "min_parking_per_household": None,
            "max_parking_per_household": None,
            "min_age": None,
            "max_age": None,
            "min_exclusive_area": None,
            "max_exclusive_area": None,
            "min_price_eok": None,
            "max_price_eok": None,
        }

    def _extract_filters(self, user_query: str) -> dict:
        client = self._get_openai_client()
        if client is None:
            raise RuntimeError("LLM client not configured")

        model = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or "gpt-4o").strip()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "apartment_filter",
                    "strict": True,
                    "schema": FILTER_SCHEMA,
                },
            },
            temperature=0,
        )
        return json.loads(completion.choices[0].message.content)

    def run(self, req: AgentRequest, ctx: AgentContext) -> AgentResponse:
        trace = ["sub_apartment_agent.run invoked", "with filter extraction + per-user history"]
        try:
            if self.agent is None:
                try:
                    self.agent = create_apartment_search_agent(csv_path=self.csv_path)
                except ModuleNotFoundError as exc:
                    return AgentResponse(
                        success=False,
                        message="아파트 추천 기능을 사용하려면 `agent_framework` 설치가 필요합니다.",
                        data={},
                        trace=trace,
                        errors=[str(exc)],
                    )

            session = self._get_session(req.user_id)
            history_text = "\n".join([f"- {q}" for q in session.user_history]) or "- (없음)"
            prompt = (
                "# Below is the history:\n"
                f"## {history_text}\n\n"
                "# User Question:\n"
                f"## {req.text}"
            ).strip()

            try:
                llm_filters = self._extract_filters(prompt)
                trace.append("filter_source=llm")
            except Exception as llm_exc:
                llm_filters = self._empty_filter_payload()
                trace.append(f"filter_source=rule_fallback({type(llm_exc).__name__})")

            # LLM 실패/누락을 보완하기 위한 규칙 파서 (명확 표현 우선 반영)
            rule_filters = self._rule_extract_filters(req.text)
            merged_filters = dict(llm_filters)
            for k, v in rule_filters.items():
                if v is not None:
                    merged_filters[k] = v

            current_specified = {k: v for k, v in merged_filters.items() if v is not None}
            if self._is_followup_only(req.text) or not current_specified:
                request_filters = dict(session.last_filters) if session.last_filters else merged_filters
                trace.append("filter_merge=use_last_filters")
            else:
                request_filters = dict(session.last_filters)
                request_filters.update(current_specified)
                trace.append("filter_merge=last_plus_current")

            pre_result = json.dumps(request_filters, ensure_ascii=False)
            result = asyncio.run(self.agent.run(pre_result))
            result_text = result.text

            session.user_history.append(pre_result)
            session.last_filters = dict(request_filters)

            return AgentResponse(success=True, message=result_text, data={}, trace=trace)
        except Exception as exc:  # noqa: BLE001
            return AgentResponse(
                success=False,
                message="아파트 추천 처리 중 오류가 발생했습니다.",
                data={},
                trace=trace,
                errors=[str(exc)],
            )

    def _rule_extract_filters(self, text: str) -> dict:
        t = text or ""
        out = self._empty_filter_payload()

        # 지역
        m_gu = re.search(r"([가-힣]+구)", t)
        if m_gu:
            out["si_gungu"] = m_gu.group(1)
        m_dong = re.search(r"([가-힣]+동)", t)
        if m_dong:
            out["eupmyeondong"] = m_dong.group(1)

        # 가격
        m_band = re.search(r"(\d+)\s*억\s*대", t)
        if m_band:
            v = float(m_band.group(1))
            out["min_price_eok"] = v
            out["max_price_eok"] = v + 1.0
        m_min = re.search(r"(\d+)\s*억\s*(이상|초과)", t)
        if m_min:
            out["min_price_eok"] = float(m_min.group(1))
        m_max = re.search(r"(\d+)\s*억\s*(이하|미만)", t)
        if m_max:
            out["max_price_eok"] = float(m_max.group(1))

        # 세대수
        m_house_min = re.search(r"(\\d{2,5})\\s*세대\\s*(이상|초과)", t)
        if m_house_min:
            out["min_households"] = int(m_house_min.group(1))
        m_house_max = re.search(r"(\\d{2,5})\\s*세대\\s*(이하|미만)", t)
        if m_house_max:
            out["max_households"] = int(m_house_max.group(1))

        # 복도유형 (부정 표현 포함)
        neg = any(k in t for k in ["싫", "제외", "빼", "말고", "아닌"])
        if "복도식" in t:
            out["corridor_type"] = "계단식" if neg else "복도식"
        if "계단식" in t:
            out["corridor_type"] = "복도식" if neg else "계단식"

        return out

    def _is_followup_only(self, text: str) -> bool:
        t = (text or "").strip().lower().replace(" ", "")
        followup = {
            "응",
            "네",
            "그래",
            "좋아",
            "다시검색",
            "다시검색해줘",
            "확인",
            "확인해줘",
            "진행",
            "진행해줘",
            "오케이",
            "ok",
            "yes",
        }
        return t in {x.lower().replace(" ", "") for x in followup}
