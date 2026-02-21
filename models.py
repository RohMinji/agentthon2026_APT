from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    user_id: str
    text: str
    timezone: str = "Asia/Seoul"
    metadata: dict[str, Any] = field(default_factory=dict)
    attachments: list[Any] = field(default_factory=list)


@dataclass
class AgentResponse:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    request_id: str
    capabilities: set[str]
    user_role: str
    env: dict[str, str] = field(default_factory=dict)
