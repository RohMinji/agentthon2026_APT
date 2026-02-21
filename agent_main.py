from __future__ import annotations

import json
import os
import re
from datetime import datetime

import telebot

from intent_router import classify_intent
from models import AgentRequest, AgentResponse
from orchestrator_core import execute_with_registry
from security_utils import mask_pii
from sub_apartment_recommend_agent import SubApartmentRecommendAgent
from sub_budget_agent import SubBudgetAgent
from sub_buying_plan_manager import SubBuyingPlanManagerAgent
from sub_qa_agent import SubQAAgent
from sub_real_estate_faq_rag_agent import SubRealEstateFaqRagAgent
from sub_real_estate_peak_agent import SubRealEstatePeakAgent

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

if load_dotenv is not None:
    load_dotenv()


AGENT_REGISTRY = {
    "apartment_recommend": SubApartmentRecommendAgent(),
    "budget_estimate": SubBudgetAgent(),
    "peak_compare": SubRealEstatePeakAgent(),
    "buying_plan": SubBuyingPlanManagerAgent(),
    "faq_rag": SubRealEstateFaqRagAgent(),
    "qa_report": SubQAAgent(),
}

PENDING_BUYING_CONFIRM: dict[str, AgentRequest] = {}
PENDING_QA_CONFIRM: dict[str, AgentRequest] = {}
LAST_INTENT_TEXT: dict[str, dict[str, str]] = {}
LAST_BOT_REPLY: dict[str, str] = {}
LAST_USER_QUERY: dict[str, str] = {}
USER_CONTEXT_HISTORY: dict[str, list[dict[str, str]]] = {}

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN 환경변수가 필요합니다.")

bot = telebot.TeleBot(TELEGRAM_TOKEN)
DEBUG_CONSOLE_LOG = (os.getenv("DEBUG_CONSOLE_LOG", "1").strip() == "1")


def _is_yes(text: str) -> bool:
    t = text.strip().lower()
    return t in {"예", "네", "응", "yes", "y", "ok", "보내", "보내줘"} or "보내" in t


def _is_no(text: str) -> bool:
    t = text.strip().lower()
    return t in {"아니", "아니오", "no", "n", "취소"} or "취소" in t


def _is_followup_confirm(text: str) -> bool:
    t = text.strip().lower()
    tokens = {
        "확인해줘",
        "확인",
        "진행해줘",
        "진행",
        "응",
        "네",
        "그래",
        "좋아",
        "오케이",
        "ok",
        "yes",
    }
    return t in tokens


def _is_peak_followup(text: str) -> bool:
    t = text.strip().lower().replace(" ", "")
    tokens = {
        "다른평형대는",
        "다른평형대",
        "다른평형",
        "다른평수",
        "다른면적",
        "평형별로",
    }
    return t in tokens or any(tok in t for tok in tokens)


def _is_help_query(text: str) -> bool:
    t = text.strip().lower()
    keywords = [
        "뭘해줄수있어",
        "뭐해줄수있어",
        "뭘할수있어",
        "뭐할수있어",
        "무얼할수있어",
        "무엇을할수있어",
        "너는무엇을할수있어",
        "너는뭘할수있어",
        "도움말",
        "사용법",
        "무엇을 할 수 있어",
        "무엇을할수있어",
        "help",
        "가능한 기능",
        "기능 알려",
        "어떤 기능",
    ]
    normalized = re.sub(r"[^0-9a-z가-힣]", "", t.replace(" ", ""))
    return any(k.replace(" ", "") in normalized for k in keywords)


def _help_message() -> str:
    return (
        "제가 도와드릴 수 있는 기능입니다.\n\n"
        "1. 아파트 추천: 조건(지역/가격/면적/난방/세대수)으로 매물 탐색\n"
        "2. 예산 추정: 월소득/보유자금 기반 대략적 매수 예산 계산\n"
        "3. 전고점 비교: 특정 단지의 전고점 대비 현재가 확인\n"
        "4. 매매 일정 정리: 계약일/잔금일 기준 필수·권장 일정 생성\n"
        "5. 정책 FAQ: PDF 근거 기반으로 정책 질의 응답\n"
        "6. QA 리포트: 민감정보 마스킹 및 QA 리포트 처리\n\n"
        "예시 질문\n"
        "- 송파구 15억 이하 84㎡ 아파트 추천해줘\n"
        "- 월소득 500, 보유자금 2억이면 예산 얼마나 돼?\n"
        "- 잠실 리센츠 전고점 대비 현재가 알려줘\n"
        "- 계약일 2026-05-10, 잔금일 2026-07-01 일정 정리해줘\n"
        "- 취득세 신고 기한을 정책 문서 근거로 알려줘"
    )


def execute_request(req: AgentRequest, intent: str | None = None) -> AgentResponse:
    return execute_with_registry(
        req,
        AGENT_REGISTRY,
        intent_override=intent,
        enabled_capabilities=None,
        user_role=os.getenv("DEFAULT_USER_ROLE", "user"),
        audit_log_path=os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl"),
    )


def _clear_user_state(user_id: str) -> None:
    PENDING_BUYING_CONFIRM.pop(user_id, None)
    PENDING_QA_CONFIRM.pop(user_id, None)
    LAST_INTENT_TEXT.pop(user_id, None)
    LAST_BOT_REPLY.pop(user_id, None)
    LAST_USER_QUERY.pop(user_id, None)
    USER_CONTEXT_HISTORY.pop(user_id, None)


def _reply_with_waiting(chat_id: int, build_reply, waiting_text: str = "🤖 AI가 생각 중입니다...") -> None:
    bot.send_chat_action(chat_id, "typing")
    waiting_msg = bot.send_message(chat_id, waiting_text)
    try:
        reply_text = build_reply()
        bot.edit_message_text(chat_id=chat_id, message_id=waiting_msg.message_id, text=reply_text)
    except Exception as e:
        bot.edit_message_text(chat_id=chat_id, message_id=waiting_msg.message_id, text=f"❌ 오류 발생: {str(e)}")


def _has_intent_keyword(text: str, intent: str) -> bool:
    t = (text or "").lower()
    if intent == "qa_report":
        qa_keys = ["qa 리포트", "qa리포트", "qa 평가", "qa평가", "qa 점검", "qa점검", "quality report"]
        if any(k in t for k in qa_keys):
            return True
        tokens = t.replace("/", " ").replace("-", " ").split()
        return "qa" in tokens

    mapping = {
        "buying_plan": ["계약일", "잔금일", "취득세", "등기", "입주", "전입"],
        "peak_compare": ["전고점", "현재가 비교", "최고가"],
        "budget_estimate": ["예산", "월소득", "대출", "dsr"],
        "faq_rag": ["정책", "faq", "pdf", "근거", "출처"],
    }
    return any(k in t for k in mapping.get(intent, []))


def _detect_intent(text: str) -> tuple[str, bool]:
    """
    GPT intent + rule intent 교차검증.
    불일치하고 신호가 약하면 uncertain=True로 반환한다.
    """
    gpt_intent = _classify_intent_with_gpt(text)
    rule_intent = classify_intent(text)

    if not gpt_intent:
        return rule_intent, False
    if gpt_intent == rule_intent:
        return gpt_intent, False

    # GPT가 특수 intent를 강하게 주장하지만 키워드 근거가 없으면 불확실 처리
    if gpt_intent in {"qa_report", "buying_plan", "peak_compare", "budget_estimate", "faq_rag"}:
        if not _has_intent_keyword(text, gpt_intent):
            return rule_intent, True

    return gpt_intent, False


def _classify_intent_with_gpt(text: str) -> str | None:
    if OpenAI is None:
        return None

    api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
    model = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or "gpt-4o").strip()
    if not (api_key and endpoint):
        return None

    try:
        client = OpenAI(base_url=f"{endpoint}/openai/v1/", api_key=api_key)
        schema = {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "apartment_recommend",
                        "budget_estimate",
                        "peak_compare",
                        "buying_plan",
                        "faq_rag",
                        "qa_report",
                    ],
                }
            },
            "required": ["intent"],
            "additionalProperties": False,
        }
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "너는 부동산 멀티에이전트 라우터다. 사용자 문장을 아래 intent 중 하나로만 분류하라. "
                        "반드시 JSON만 출력한다."
                    ),
                },
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "intent_router", "strict": True, "schema": schema},
            },
            temperature=0,
        )
        payload = json.loads(completion.choices[0].message.content)
        intent = payload.get("intent")
        if intent in AGENT_REGISTRY:
            return intent
    except Exception:
        return None
    return None


def _extract_household_suggestion(reply_text: str) -> str:
    m = re.search(r"(\\d{2,5})\\s*세대\\s*이상", reply_text or "")
    if not m:
        return ""
    return f"{m.group(1)}세대 이상"


def _looks_like_pii_mask_test(text: str) -> bool:
    t = (text or "").lower()
    markers = [
        "[phone_masked]",
        "[email_masked]",
        "[rrn_masked]",
        "[account_masked]",
        "연락처",
        "주민번호",
        "계좌",
        "이메일",
    ]
    return any(m in t for m in markers)


def _append_context_history(user_id: str, user_text: str, intent: str, bot_text: str) -> None:
    history = USER_CONTEXT_HISTORY.setdefault(user_id, [])
    history.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "intent": intent,
            "user": user_text,
            "bot": bot_text,
        }
    )


@bot.message_handler(commands=["new", "reset"])
def reset_cmd(message):
    user_id = str(message.from_user.id)
    _clear_user_state(user_id)
    apt_agent = AGENT_REGISTRY.get("apartment_recommend")
    if apt_agent and hasattr(apt_agent, "reset_user_session"):
        try:
            apt_agent.reset_user_session(user_id)
        except Exception:
            pass
    bot.send_message(message.chat.id, "상태를 초기화했습니다.")


@bot.message_handler(func=lambda m: True)
def handle_message(message):
    user_id = str(message.from_user.id)
    text = (message.text or "").strip()
    if not text:
        bot.send_message(message.chat.id, "질문을 입력해주세요.")
        return

    if _is_help_query(text):
        msg = _help_message()
        LAST_BOT_REPLY[user_id] = msg
        _append_context_history(user_id, text, "help", msg)
        if DEBUG_CONSOLE_LOG:
            print(f"[USER:{user_id}] {mask_pii(text)}\n[INTENT] help\n[BOT] {mask_pii(msg)}\n")
        bot.send_message(message.chat.id, msg)
        return

    pending = PENDING_BUYING_CONFIRM.get(user_id)
    if pending is not None:
        if _is_yes(text):
            pending.metadata["confirm_send"] = True
            def _on_confirm_yes() -> str:
                resp = execute_request(pending, intent="buying_plan")
                PENDING_BUYING_CONFIRM.pop(user_id, None)
                _log_exchange(user_id, pending.text, "buying_plan", resp)
                msg = _format_for_telegram(resp)
                LAST_BOT_REPLY[user_id] = msg
                _append_context_history(user_id, pending.text, "buying_plan", msg)
                return msg

            _reply_with_waiting(message.chat.id, _on_confirm_yes)
            return
        if _is_no(text):
            PENDING_BUYING_CONFIRM.pop(user_id, None)
            cancel_msg = "메일 발송을 취소했습니다."
            _append_context_history(user_id, text, "buying_plan", cancel_msg)
            bot.send_message(message.chat.id, cancel_msg)
            return
        guide_msg = "메일 발송 여부를 '예/아니오'로 답해주세요."
        _append_context_history(user_id, text, "buying_plan", guide_msg)
        bot.send_message(message.chat.id, guide_msg)
        return

    pending_qa = PENDING_QA_CONFIRM.get(user_id)
    if pending_qa is not None:
        if _is_yes(text):
            pending_qa.metadata["send_mail"] = True
            if not pending_qa.metadata.get("assistant_answer"):
                pending_qa.metadata["assistant_answer"] = LAST_BOT_REPLY.get(user_id, "")

            def _on_qa_confirm_yes() -> str:
                resp = execute_request(pending_qa, intent="qa_report")
                PENDING_QA_CONFIRM.pop(user_id, None)
                _log_exchange(user_id, pending_qa.text, "qa_report", resp)
                msg = _format_for_telegram(resp)
                LAST_BOT_REPLY[user_id] = msg
                _append_context_history(user_id, pending_qa.text, "qa_report", msg)
                return msg

            _reply_with_waiting(message.chat.id, _on_qa_confirm_yes)
            return
        if _is_no(text):
            PENDING_QA_CONFIRM.pop(user_id, None)
            cancel_msg = "QA 리포트 메일 발송을 취소했습니다."
            _append_context_history(user_id, text, "qa_report", cancel_msg)
            bot.send_message(message.chat.id, cancel_msg)
            return
        guide_msg = "QA 리포트 메일 발송 여부를 '예/아니오'로 답해주세요."
        _append_context_history(user_id, text, "qa_report", guide_msg)
        bot.send_message(message.chat.id, guide_msg)
        return

    req = AgentRequest(user_id=user_id, text=text)
    intent, uncertain_intent = _detect_intent(text)

    # 하드 가드: 개인정보/마스킹 테스트 문장은 QA 리포트로 보내지 않음
    if _looks_like_pii_mask_test(text) and not _has_intent_keyword(text, "qa_report"):
        uncertain_intent = True

    # 하드 가드: QA 리포트는 명시 키워드가 있을 때만 허용
    if intent == "qa_report" and not _has_intent_keyword(text, "qa_report"):
        uncertain_intent = True

    if uncertain_intent:
        clarify = (
            "요청 의도를 정확히 파악하기 어려워서 확인이 필요해요.\n"
            "아래 중 원하는 작업을 짧게 알려주세요.\n"
            "- 아파트 추천\n"
            "- 전고점 비교\n"
            "- 매매 일정 생성\n"
            "- 정책 FAQ\n"
            "- QA 리포트"
        )
        LAST_BOT_REPLY[user_id] = clarify
        _append_context_history(user_id, text, "uncertain", clarify)
        if DEBUG_CONSOLE_LOG:
            print(f"[USER:{user_id}] {mask_pii(text)}\n[INTENT] uncertain\n[BOT] {mask_pii(clarify)}\n")
        bot.send_message(message.chat.id, clarify)
        return

    # "다른 평형대는?" 같은 후속 질문은 intent가 흔들려도 peak_compare로 강제 연결
    if _is_peak_followup(text):
        prev_peak = LAST_INTENT_TEXT.get(user_id, {}).get("peak_compare")
        if prev_peak:
            intent = "peak_compare"
            req.text = prev_peak + " 다른 평형대"

    if intent == "apartment_recommend" and _is_followup_confirm(text):
        prev = LAST_INTENT_TEXT.get(user_id, {}).get("apartment_recommend")
        if prev:
            addon = _extract_household_suggestion(LAST_BOT_REPLY.get(user_id, ""))
            req.text = f"{prev} {addon}".strip()

    if intent == "buying_plan":
        def _on_buying_plan() -> str:
            req.metadata["confirm_send"] = False
            resp = execute_request(req, intent=intent)
            if resp.success:
                _log_exchange(user_id, text, intent, resp)
                PENDING_BUYING_CONFIRM[user_id] = req
                msg = _format_for_telegram(resp) + "\n\n메일로 보내드릴까요? (예/아니오)"
                LAST_BOT_REPLY[user_id] = msg
                return msg
            msg = _format_for_telegram(resp)
            LAST_BOT_REPLY[user_id] = msg
            return msg

        _reply_with_waiting(message.chat.id, _on_buying_plan)
        return

    if intent == "qa_report":
        wants_send = ("보내" in text) or ("메일" in text) or ("send" in text.lower())
        req.metadata["send_mail"] = wants_send
        req.metadata["assistant_answer"] = LAST_BOT_REPLY.get(user_id, "")
        req.metadata["target_question"] = LAST_USER_QUERY.get(user_id, text)
        req.metadata["context_history"] = USER_CONTEXT_HISTORY.get(user_id, [])

        def _on_qa() -> str:
            resp = execute_request(req, intent=intent)
            _log_exchange(user_id, text, intent, resp)
            msg = _format_for_telegram(resp)
            if resp.success and not wants_send:
                PENDING_QA_CONFIRM[user_id] = AgentRequest(
                    user_id=req.user_id,
                    text=req.text,
                    timezone=req.timezone,
                    metadata={
                        "send_mail": False,
                        "assistant_answer": LAST_BOT_REPLY.get(user_id, ""),
                        "target_question": LAST_USER_QUERY.get(user_id, text),
                        "context_history": USER_CONTEXT_HISTORY.get(user_id, []),
                    },
                    attachments=req.attachments,
                )
                msg = msg + "\n\nQA 리포트를 메일로도 보내드릴까요? (예/아니오)"
            LAST_BOT_REPLY[user_id] = msg
            _append_context_history(user_id, text, intent, msg)
            return msg

        _reply_with_waiting(message.chat.id, _on_qa, waiting_text="리포트를 생성중입니다. 잠시만 기다려주세요")
        return

    def _on_general() -> str:
        resp = execute_request(req, intent=intent)
        if intent == "apartment_recommend" and not _is_followup_confirm(text):
            LAST_INTENT_TEXT.setdefault(user_id, {})["apartment_recommend"] = text
        if intent == "peak_compare" and not _is_peak_followup(text):
            LAST_INTENT_TEXT.setdefault(user_id, {})["peak_compare"] = req.text
        _log_exchange(user_id, text, intent, resp)
        msg = _format_for_telegram(resp)
        LAST_BOT_REPLY[user_id] = msg
        _append_context_history(user_id, text, intent, msg)
        if intent != "qa_report":
            LAST_USER_QUERY[user_id] = text
        return msg

    _reply_with_waiting(message.chat.id, _on_general)


def _format_for_telegram(resp: AgentResponse) -> str:
    lines = [resp.message]
    if resp.errors:
        lines.append("")
        lines.append("[errors]")
        for e in resp.errors:
            lines.append(f"- {e}")
    return "\n".join(lines)


def _log_exchange(user_id: str, user_text: str, intent: str, resp: AgentResponse) -> None:
    if not DEBUG_CONSOLE_LOG:
        return
    safe_user_text = mask_pii(user_text)
    safe_reply = mask_pii(resp.message)
    print(
        f"[USER:{user_id}] {safe_user_text}\n"
        f"[INTENT] {intent}\n"
        f"[BOT] {safe_reply}\n"
    )


if __name__ == "__main__":
    print("APT 에이전트 실행 중...")
    bot.infinity_polling()
