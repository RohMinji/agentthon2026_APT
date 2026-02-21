from __future__ import annotations

import csv
import json
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

from models import AgentContext, AgentRequest, AgentResponse
from security_utils import BANK_ACCOUNT_PATTERN, DETAIL_ADDR_PATTERN, EMAIL_PATTERN, PHONE_PATTERN, RRN_PATTERN, mask_pii
from sub_agent_base import SubAgentBase

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


BEST_ANSWERS_CSV = os.getenv("BEST_ANSWERS_CSV", "best_answers.csv")


@dataclass
class EvalResult:
    score: int
    verdict: str
    reason: str
    matched_question: str
    similarity: float
    llm_status: str


class SubQAAgent(SubAgentBase):
    name = "sub_qa_agent"
    description = "best_answers 기반 QA 평가 + 메일 리포트"
    capabilities = {"email_send"}

    def __init__(self, best_answers_csv: str = BEST_ANSWERS_CSV):
        self.best_answers_csv = Path(best_answers_csv)
        self.best_rows = self._load_best_rows()

    def run(self, req: AgentRequest, ctx: AgentContext) -> AgentResponse:
        trace = ["sub_qa_agent.run invoked", "best-answer matching + llm scoring"]

        target_answer = str(req.metadata.get("assistant_answer") or "").strip()
        if not target_answer:
            return AgentResponse(
                success=False,
                message="QA 평가할 답변이 없습니다. 먼저 평가 대상 답변이 있어야 합니다.",
                data={},
                trace=trace,
                errors=["missing_assistant_answer"],
            )

        user_question = str(req.metadata.get("target_question") or req.text or "").strip()
        ev = self._evaluate(user_question=user_question, assistant_answer=target_answer)
        context_history = req.metadata.get("context_history") or []

        masked_q = mask_pii(user_question)
        masked_a = mask_pii(target_answer)
        security = self._security_eval(user_question=user_question, assistant_answer=target_answer, context_history=context_history)
        context_log_path = self._write_context_log(context_history)
        report = (
            "QA 평가 리포트\n"
            f"- timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
            "\n[답변 퀄리티 평가]\n"
            f"- score: {ev.score}\n"
            f"- verdict: {ev.verdict}\n"
            f"- matched_question: {ev.matched_question}\n"
            f"- similarity: {ev.similarity:.3f}\n"
            f"- llm_status: {ev.llm_status}\n"
            f"- reason: {ev.reason}\n"
            "\n[보안 평가]\n"
            f"- pii_detected_count: {security['pii_detected_count']}\n"
            f"- pii_masked_count: {security['pii_masked_count']}\n"
            f"- security_verdict: {security['security_verdict']}\n"
            f"- security_reason: {security['security_reason']}\n"
            "\n[입력]\n"
            f"- question: {masked_q}\n"
            f"- answer: {masked_a[:1500]}"
        )

        sent = False
        if req.metadata.get("send_mail", False):
            if "email_send" not in ctx.capabilities:
                return AgentResponse(
                    success=False,
                    message="메일 발송 권한이 없어 QA 메일을 보낼 수 없습니다.",
                    data={"score": ev.score, "verdict": ev.verdict, "report": report},
                    trace=trace,
                    errors=["capability_blocked:email_send"],
                )
            try:
                _send_mail(
                    subject=f"[QA Eval] score={ev.score} {ev.verdict}",
                    body=report,
                    attachments=[context_log_path],
                )
                sent = True
                trace.append("qa eval report emailed")
            except Exception as exc:  # noqa: BLE001
                return AgentResponse(
                    success=False,
                    message="QA 평가는 완료했지만 메일 발송에 실패했습니다.",
                    data={"score": ev.score, "verdict": ev.verdict, "report": report},
                    trace=trace,
                    errors=[str(exc)],
                )

        msg = "QA 평가 리포트를 생성했습니다." + (" 메일 발송까지 완료했습니다." if sent else "")
        return AgentResponse(
            success=True,
            message=msg,
            data={
                "score": ev.score,
                "verdict": ev.verdict,
                "reason": ev.reason,
                "matched_question": ev.matched_question,
                "similarity": ev.similarity,
                "llm_status": ev.llm_status,
                "security_verdict": security["security_verdict"],
                "security_reason": security["security_reason"],
                "context_log_path": str(context_log_path),
                "email_sent": sent,
                "report": report,
            },
            trace=trace,
        )

    def _load_best_rows(self) -> list[dict[str, str]]:
        if not self.best_answers_csv.exists():
            return []
        rows: list[dict[str, str]] = []
        with open(self.best_answers_csv, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("question") or "").strip()
                a = (row.get("excellent_answer") or "").strip()
                if q and a:
                    rows.append({"question": q, "excellent_answer": a})
        return rows

    def _best_gold(self, question: str) -> tuple[Optional[dict[str, str]], float]:
        if not self.best_rows:
            return None, 0.0
        q = question or ""
        scored = []
        for row in self.best_rows:
            sim = SequenceMatcher(None, q, row["question"]).ratio()
            scored.append((sim, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1], scored[0][0]

    def _evaluate(self, user_question: str, assistant_answer: str) -> EvalResult:
        gold, q_sim = self._best_gold(user_question)
        if gold is None:
            return EvalResult(0, "REVIEW", "best_answers.csv를 찾을 수 없거나 비어 있습니다.", "", 0.0, "not_used")

        if q_sim < 0.2:
            return EvalResult(
                0,
                "REVIEW",
                "유사한 기준 질문을 찾지 못해 신뢰 가능한 평가를 보류했습니다.",
                gold["question"],
                q_sim,
                "not_used",
            )

        parsed, llm_err = self._evaluate_with_llm(user_question, gold["excellent_answer"], assistant_answer)
        if parsed is None:
            # fallback heuristic
            a_sim = SequenceMatcher(None, assistant_answer, gold["excellent_answer"]).ratio()
            score = int(round(a_sim * 100))
            verdict = "PASS" if score >= 70 else "REVIEW"
            return EvalResult(
                score,
                verdict,
                f"LLM 평가 실패로 유사도 기반 대체 평가 ({llm_err})",
                gold["question"],
                q_sim,
                f"failed:{llm_err}",
            )

        score = max(0, min(100, int(parsed.get("score", 0))))
        verdict = str(parsed.get("verdict", "REVIEW")).upper()
        if verdict not in {"PASS", "REVIEW"}:
            verdict = "REVIEW"
        reason = str(parsed.get("reason", ""))[:500]
        return EvalResult(score, verdict, reason, gold["question"], q_sim, "ok")

    def _evaluate_with_llm(self, user_question: str, excellent_answer: str, assistant_answer: str) -> tuple[Optional[dict[str, Any]], str]:
        if OpenAI is None:
            return None, "openai_not_installed"

        api_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
        endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip().rstrip("/")
        model = (os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or "gpt-4o").strip()
        if not (api_key and endpoint):
            return None, "azure_env_missing"

        try:
            client = OpenAI(base_url=f"{endpoint}/openai/V1/", api_key=api_key)
            schema = {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["PASS", "REVIEW"]},
                    "reason": {"type": "string"},
                },
                "required": ["score", "verdict", "reason"],
                "additionalProperties": False,
            }
            prompt = (
                "[사용자 질문]\n"
                f"{user_question}\n\n"
                "[모범 답변]\n"
                f"{excellent_answer}\n\n"
                "[평가 대상 답변]\n"
                f"{assistant_answer}\n\n"
                "정확성/충실성/실행가능성/명확성 기준으로 평가하라."
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "너는 QA 평가자다. 반드시 JSON만 출력한다."},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "qa_eval", "strict": True, "schema": schema},
                },
                temperature=0,
            )
            return json.loads(completion.choices[0].message.content), ""
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def _security_eval(self, user_question: str, assistant_answer: str, context_history: list[dict]) -> dict[str, Any]:
        raw = "\n".join([user_question, assistant_answer] + [str(x) for x in context_history])
        masked = mask_pii(raw)
        detected = self._count_pii_patterns(raw)
        masked_detected = self._count_pii_patterns(masked)
        verdict = "PASS" if masked_detected == 0 else "REVIEW"
        reason = "민감정보 패턴 마스킹 완료" if verdict == "PASS" else "마스킹 후에도 민감정보 패턴 일부 잔존"
        return {
            "pii_detected_count": detected,
            "pii_masked_count": masked_detected,
            "security_verdict": verdict,
            "security_reason": reason,
        }

    def _count_pii_patterns(self, text: str) -> int:
        return (
            len(PHONE_PATTERN.findall(text or ""))
            + len(EMAIL_PATTERN.findall(text or ""))
            + len(RRN_PATTERN.findall(text or ""))
            + len(DETAIL_ADDR_PATTERN.findall(text or ""))
            + len(BANK_ACCOUNT_PATTERN.findall(text or ""))
        )

    def _write_context_log(self, context_history: list[dict]) -> Path:
        lines = ["[Masked Conversation Context]"]
        if not context_history:
            lines.append("No conversation history provided.")
        for idx, item in enumerate(context_history, start=1):
            q = mask_pii(str(item.get("user", "")))
            a = mask_pii(str(item.get("bot", "")))
            intent = str(item.get("intent", ""))
            ts = str(item.get("timestamp", ""))
            lines.append(f"{idx}. timestamp={ts} intent={intent}")
            lines.append(f"   user: {q}")
            lines.append(f"   bot: {a}")
        path = Path("context_log.txt")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def _send_mail(subject: str, body: str, attachments: Optional[list[Path]] = None) -> None:
    user = (os.getenv("GMAIL_SMTP_USER") or "").strip()
    app_pw = (os.getenv("GMAIL_APP_PASSWORD") or "").strip()
    to = (os.getenv("GMAIL_TO") or "").strip()
    if not (user and app_pw and to):
        raise RuntimeError("메일 설정이 준비되지 않았습니다.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)
    for path in attachments or []:
        data = path.read_bytes()
        msg.add_attachment(data, maintype="text", subtype="plain", filename=path.name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, app_pw)
        server.send_message(msg)
