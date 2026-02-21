from __future__ import annotations

from abc import ABC, abstractmethod

from models import AgentContext, AgentRequest, AgentResponse


class SubAgentBase(ABC):
    name: str = ""
    description: str = ""
    capabilities: set[str] = set()

    @abstractmethod
    def run(self, req: AgentRequest, ctx: AgentContext) -> AgentResponse:
        raise NotImplementedError
