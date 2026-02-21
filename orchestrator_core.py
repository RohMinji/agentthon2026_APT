from __future__ import annotations

import os
from uuid import uuid4
from typing import Optional

from intent_router import classify_intent
from models import AgentContext, AgentRequest, AgentResponse
from security_utils import append_audit_log
from sub_agent_base import SubAgentBase


def parse_capabilities(value: Optional[str]) -> set[str]:
    if not value:
        return {"email_send", "pdf_read", "web_search"}
    return {x.strip() for x in value.split(",") if x.strip()}


def ensure_capabilities(agent_caps: set[str], ctx_caps: set[str]) -> tuple[bool, list[str]]:
    missing = sorted(agent_caps - ctx_caps)
    return (len(missing) == 0, missing)


def execute_with_registry(
    req: AgentRequest,
    registry: dict[str, SubAgentBase],
    *,
    intent_override: Optional[str] = None,
    enabled_capabilities: Optional[set[str]] = None,
    user_role: str = "user",
    audit_log_path: Optional[str] = None,
) -> AgentResponse:
    request_id = str(uuid4())
    intent = intent_override or classify_intent(req.text)
    agent = registry[intent]

    ctx = AgentContext(
        request_id=request_id,
        capabilities=enabled_capabilities if enabled_capabilities is not None else parse_capabilities(os.getenv("ENABLED_CAPABILITIES")),
        user_role=user_role,
        env={"AUDIT_LOG_PATH": audit_log_path or os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")},
    )

    ok, missing = ensure_capabilities(agent.capabilities, ctx.capabilities)
    if not ok:
        response = AgentResponse(
            success=False,
            message="권한(capability) 정책에 의해 요청이 차단되었습니다.",
            data={"intent": intent, "missing_capabilities": missing},
            trace=[f"intent={intent}", f"blocked_missing={','.join(missing)}"],
            errors=[f"capability_blocked:{m}" for m in missing],
        )
    else:
        response = agent.run(req, ctx)
        response.trace.insert(0, f"intent={intent}")
        response.data.setdefault("intent", intent)

    append_audit_log(
        request_id=request_id,
        user_id=req.user_id,
        selected_agent=agent.name,
        input_text=req.text,
        output_text=response.message,
        log_path=ctx.env.get("AUDIT_LOG_PATH"),
    )
    return response
