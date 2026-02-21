from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from models import AgentContext, AgentRequest, AgentResponse
from sub_agent_base import SubAgentBase


SAFE_GUARD_LINE = "정확한 답변을 위해 관련 문서 및 근거를 확인했습니다."


@dataclass
class Chunk:
    text: str
    source: str
    page: int


class SubRealEstateFaqRagAgent(SubAgentBase):
    name = "sub_real_estate_faq_rag_agent"
    description = "PDF 근거 기반 FAQ RAG"
    capabilities = {"pdf_read"}

    def __init__(self, pdf_paths: Optional[list[str]] = None):
        env_paths = os.getenv("FAQ_PDF_PATHS", "DATA/real_estate_policy_info.pdf,DATA/real_estate_news.pdf")
        self.pdf_paths = [p.strip() for p in (pdf_paths or env_paths.split(",")) if p.strip()]
        self.chunks: list[Chunk] = []
        self._loaded = False

    def run(self, req: AgentRequest, ctx: AgentContext) -> AgentResponse:
        trace = ["sub_real_estate_faq_rag_agent.run invoked"]

        if "pdf_read" not in ctx.capabilities:
            return AgentResponse(False, "문서 조회 권한이 없어 답변할 수 없습니다.", {}, trace, errors=["capability_blocked:pdf_read"])

        self._ensure_loaded()
        trace.append(f"chunk_count={len(self.chunks)}")

        top = self._search(req.text, top_k=3)
        citations = [
            {"source": c.source, "page": c.page, "snippet": c.text[:180]}
            for c, score in top
            if score > 0
        ]

        if citations:
            answer = self._compose_answer(req.text, [c for c, _ in top if _ > 0])
            message = f"{SAFE_GUARD_LINE}\n\n{answer}"
            return AgentResponse(True, message, data={"mode": "pdf_rag"}, trace=trace, citations=citations)

        if "web_search" not in ctx.capabilities:
            return AgentResponse(
                False,
                f"{SAFE_GUARD_LINE}\n\n문서 근거를 찾지 못해 답변을 보류합니다.",
                {},
                trace,
                errors=["no_citation"],
            )

        verified = self._verify_claim_three_sources(req.metadata)
        trace.append(f"external_verified={verified}")
        if not verified:
            return AgentResponse(
                False,
                f"{SAFE_GUARD_LINE}\n\n문서 근거 및 외부 3중 검증이 부족해 답변을 거절합니다.",
                {},
                trace,
                errors=["verification_failed"],
            )

        return AgentResponse(
            True,
            f"{SAFE_GUARD_LINE}\n\nPDF 근거는 없지만, 독립 출처 3회 이상 검증된 정보만 요약 제공합니다.",
            data={"mode": "external_verified"},
            trace=trace,
            citations=req.metadata.get("external_sources", []),
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        chunks: list[Chunk] = []
        for path in self.pdf_paths:
            p = Path(path)
            if not p.exists():
                continue
            pages = _read_pdf_pages(p)
            for page_no, text in enumerate(pages, start=1):
                clean = _sanitize_pdf_text(text)
                for segment in _chunk_text(clean, chunk_size=600, overlap=120):
                    chunks.append(Chunk(text=segment, source=str(p), page=page_no))
        self.chunks = chunks
        self._loaded = True

    def _search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q_tokens = _tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk in self.chunks:
            c_tokens = _tokenize(chunk.text)
            score = _cosine_similarity(q_tokens, c_tokens)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _compose_answer(self, question: str, chunks: list[Chunk]) -> str:
        lines = [f"질문: {question}", "근거 기반 요약:"]
        for idx, c in enumerate(chunks, start=1):
            lines.append(f"{idx}. {c.text[:180]}...")
        return "\n".join(lines)

    def _verify_claim_three_sources(self, metadata: dict[str, Any]) -> bool:
        sources = metadata.get("external_sources", []) if isinstance(metadata, dict) else []
        domains = set()
        for src in sources:
            if isinstance(src, dict):
                url = src.get("url", "")
            else:
                url = str(src)
            dom = urlparse(url).netloc.lower()
            if dom:
                domains.add(dom)
        return len(domains) >= 3


def _read_pdf_pages(path: Path) -> list[str]:
    # Supports pypdf first, then PyPDF2 fallback.
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            return [page.extract_text() or "" for page in reader.pages]
        except Exception:
            return []


def _sanitize_pdf_text(text: str) -> str:
    lowered = (text or "").lower()
    # Prompt injection defense: ignore instruction-like lines from documents.
    deny_words = ["ignore previous", "system prompt", "developer message", "지시를 무시"]
    lines = []
    for line in (text or "").splitlines():
        ll = line.lower()
        if any(dw in ll for dw in deny_words):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    if any(dw in lowered for dw in deny_words):
        cleaned += "\n[주의] 문서 내 지시문 형태 텍스트는 무시 처리됨"
    return cleaned


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    norm = re.sub(r"\s+", " ", text).strip()
    if not norm:
        return []
    chunks: list[str] = []
    start = 0
    n = len(norm)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(norm[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _tokenize(text: str) -> dict[str, float]:
    tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", text.lower())
    freq: dict[str, float] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0.0) + 1.0
    return freq


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
