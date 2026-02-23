from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import faiss # pip install pypdf sentence-transformers faiss-cpu rank-bm25
import numpy as np
from pydantic import Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from agent_framework import ai_function
except ImportError:
    def ai_function(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


try:
    from kiwipiepy import Kiwi
except ImportError:
    Kiwi = None


class KoreanTokenizer:
    """Korean tokenizer for BM25. Uses Kiwi if available, else regex fallback."""

    def __init__(self):
        self._kiwi = Kiwi() if Kiwi is not None else None

    def __getstate__(self) -> dict:
        # Avoid pickling native Kiwi object directly.
        return {"_kiwi": None}

    def __setstate__(self, state: dict) -> None:
        _ = state
        self._kiwi = Kiwi() if Kiwi is not None else None

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        if self._kiwi is not None:
            # Keep meaningful content words. Drop particles/endings to improve lexical search.
            tokens: list[str] = []
            for tok in self._kiwi.tokenize(text):
                if tok.tag.startswith(("NN", "VV", "VA", "SL", "SN")):
                    tokens.append(tok.form.lower())
            if tokens:
                return tokens

        return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


@dataclass
class Chunk:
    chunk_id: int
    source_file: str
    page: int
    text: str


class HybridPdfRAGTool:
    def __init__(
        self,
        pdf_dir: str | Path,
        index_dir: str | Path = "./knowledge/pdf_hybrid_index",
        embedding_model_name: str = "intfloat/multilingual-e5-base",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = SentenceTransformer(self.embedding_model_name)
        self.tokenizer = KoreanTokenizer()

        self.chunks: list[Chunk] = []
        self.chunk_texts: list[str] = []
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None

        self.bm25: BM25Okapi | None = None
        self.bm25_corpus_tokens: list[list[str]] = []

    def _split_text(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        out = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            out.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return out

    def _load_pdf_chunks(self) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        cid = 0

        pdf_files = sorted(self.pdf_dir.rglob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in: {self.pdf_dir}")

        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                pieces = self._split_text(raw)
                for piece in pieces:
                    all_chunks.append(
                        Chunk(
                            chunk_id=cid,
                            source_file=str(pdf_path.name),
                            page=page_idx,
                            text=piece,
                        )
                    )
                    cid += 1
        return all_chunks

    def _embed_passages(self, texts: list[str]) -> np.ndarray:
        inputs = [f"passage: {t}" for t in texts]
        vec = self.model.encode(
            inputs,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype("float32")

    def _embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [f"query: {query}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.astype("float32")

    def build_index(self, force_rebuild: bool = False) -> dict:
        self.index_dir.mkdir(parents=True, exist_ok=True)

        chunks_json = self.index_dir / "chunks.json"
        emb_npy = self.index_dir / "embeddings.npy"
        faiss_file = self.index_dir / "dense.faiss"
        bm25_file = self.index_dir / "bm25.pkl"

        if (
            not force_rebuild
            and chunks_json.exists()
            and emb_npy.exists()
            and faiss_file.exists()
            and bm25_file.exists()
        ):
            self.chunks = [Chunk(**x) for x in json.loads(chunks_json.read_text(encoding="utf-8"))]
            self.chunk_texts = [c.text for c in self.chunks]
            self.embeddings = np.load(emb_npy)
            self.index = faiss.read_index(str(faiss_file))
            with open(bm25_file, "rb") as f:
                loaded = pickle.load(f)
                # Backward compatibility:
                # old format: (bm25_obj, corpus_tokens)
                # new format: corpus_tokens only
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    self.bm25_corpus_tokens = loaded[1]
                else:
                    self.bm25_corpus_tokens = loaded
            self.bm25 = BM25Okapi(self.bm25_corpus_tokens)
            return {"status": "loaded", "chunk_count": len(self.chunks)}

        self.chunks = self._load_pdf_chunks()
        self.chunk_texts = [c.text for c in self.chunks]

        self.embeddings = self._embed_passages(self.chunk_texts)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        self.bm25_corpus_tokens = [self.tokenizer.tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

        chunks_json.write_text(
            json.dumps([c.__dict__ for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(emb_npy, self.embeddings)
        faiss.write_index(self.index, str(faiss_file))
        with open(bm25_file, "wb") as f:
            pickle.dump(self.bm25_corpus_tokens, f)

        return {"status": "built", "chunk_count": len(self.chunks)}

    def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 30,
        bm25_k: int = 30,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ) -> list[dict]:
        if self.index is None or self.bm25 is None:
            self.build_index(force_rebuild=False)

        qv = self._embed_query(query)
        dense_scores, dense_ids = self.index.search(qv, dense_k)
        dense_rank: dict[int, int] = {}
        for rank, idx in enumerate(dense_ids[0].tolist(), start=1):
            if idx >= 0:
                dense_rank[idx] = rank

        q_tokens = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ids = np.argsort(bm25_scores)[::-1][:bm25_k]
        bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_ids.tolist(), start=1)}

        # Reciprocal Rank Fusion
        rrf_k = 60
        candidate_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        fused: list[tuple[int, float]] = []
        for cid in candidate_ids:
            s_dense = dense_weight * (1.0 / (rrf_k + dense_rank[cid])) if cid in dense_rank else 0.0
            s_bm25 = bm25_weight * (1.0 / (rrf_k + bm25_rank[cid])) if cid in bm25_rank else 0.0
            fused.append((cid, s_dense + s_bm25))

        fused.sort(key=lambda x: x[1], reverse=True)
        top = fused[:top_k]

        results = []
        for cid, score in top:
            c = self.chunks[cid]
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "source_file": c.source_file,
                    "page": c.page,
                    "score": float(score),
                    "text": c.text,
                }
            )
        return results

    @ai_function(
        name="search_pdf_hybrid",
        description="PDF 벡터DB + BM25 하이브리드 검색으로 질의 관련 근거를 반환한다.",
    )
    def search_pdf_hybrid(
        self,
        query: Annotated[str, Field(description="사용자 질문")],
        top_k: Annotated[int, Field(description="반환할 근거 개수", ge=1, le=10)] = 5,
    ) -> str:
        hits = self._hybrid_search(query=query, top_k=top_k)
        payload = {
            "query": query,
            "top_k": top_k,
            "hit_count": len(hits),
            "hits": hits,
        }
        return json.dumps(payload, ensure_ascii=False)

    @ai_function(
        name="rebuild_pdf_hybrid_index",
        description="PDF 변경 시 하이브리드 인덱스를 재생성한다.",
    )
    def rebuild_pdf_hybrid_index(
        self,
        force_rebuild: Annotated[bool, Field(description="true면 강제 재생성")] = True,
    ) -> str:
        info = self.build_index(force_rebuild=force_rebuild)
        return json.dumps(info, ensure_ascii=False)


def create_pdf_rag_agent(
    pdf_dir: str | Path,
    index_dir: str | Path = "./knowledge/pdf_hybrid_index",
    credential=None,
):
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity import AzureCliCredential

    rag_tool = HybridPdfRAGTool(pdf_dir=pdf_dir, index_dir=index_dir)
    rag_tool.build_index(force_rebuild=False)

    return AzureOpenAIChatClient(
        credential=credential or AzureCliCredential()
    ).as_agent(
        name="PolicyPdfRAG",
        instructions=(
            "너는 PDF 기반 정책/규정 질의응답 전문가다. "
            "반드시 먼저 search_pdf_hybrid 도구를 호출해 근거를 수집한 뒤 답변하라. "
            "답변에는 핵심 결론을 먼저 쓰고, 근거 출처를 [파일명 p.페이지] 형식으로 명시하라. "
            "근거가 불충분하면 추측하지 말고 '근거 부족'이라고 말한 뒤 추가 확인 항목을 제시하라."
        ),
        tools=[rag_tool.search_pdf_hybrid, rag_tool.rebuild_pdf_hybrid_index],
    )
