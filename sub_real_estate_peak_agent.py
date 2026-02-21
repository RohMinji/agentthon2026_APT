from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from models import AgentContext, AgentRequest, AgentResponse
from sub_agent_base import SubAgentBase


@dataclass
class PeakInfo:
    latest_price: int
    peak_price: int
    gap_ratio: float
    area_m2: int


class SubRealEstatePeakAgent(SubAgentBase):
    name = "sub_real_estate_peak_agent"
    description = "전고점 대비 현재가 비교"
    capabilities = set()

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = Path(csv_path or os.getenv("REAL_ESTATE_CSV_PATH", "DATA/real_estate.csv"))
        self.naver_articles_path = Path(os.getenv("NAVER_ARTICLES_CSV_PATH", "DATA/naver_articles.csv"))

    def run(self, req: AgentRequest, ctx: AgentContext) -> AgentResponse:
        trace = ["sub_real_estate_peak_agent.run invoked", f"csv={self.csv_path}"]
        if not self.csv_path.exists():
            return AgentResponse(False, "거래 데이터 파일을 찾을 수 없습니다.", {}, trace, errors=[str(self.csv_path)])

        apt_name = _extract_apt_name(req.text)
        if not apt_name:
            return AgentResponse(False, "아파트명을 함께 입력해주세요. 예: 잠실 리센츠 전고점", {}, trace)

        resolved_name = _resolve_apt_name(self.csv_path, apt_name)
        trace.append(f"resolved_apt_name={resolved_name or 'N/A'}")
        if not resolved_name:
            return AgentResponse(False, f"'{apt_name}'에 대한 거래 데이터를 찾지 못했습니다.", {}, trace)

        wants_other_areas = _wants_other_areas(req.text)
        target_area = _extract_target_area(req.text)
        if target_area is not None:
            trace.append(f"target_area={target_area}㎡")
        if wants_other_areas:
            multi = _calc_peak_multi_areas(self.csv_path, resolved_name, top_n=6)
            if not multi:
                return AgentResponse(False, f"'{apt_name}'에 대한 거래 데이터를 찾지 못했습니다.", {}, trace)

            lines = [f"{resolved_name} 평형별 전고점 비교"]
            rows = []
            for area_m2, info in multi:
                current_price, source = _current_price_from_naver_listings(
                    self.naver_articles_path,
                    resolved_name,
                    area_m2,
                )
                if current_price is None:
                    current_price = info.latest_price
                    source = "실거래 최신가(폴백)"
                gap_ratio = (current_price / info.peak_price) * 100 if info.peak_price else 0.0
                lines.append(
                    f"- {area_m2}㎡형 | 현재가 {current_price:,}만원 | 전고점 {info.peak_price:,}만원 | {gap_ratio:.1f}%"
                )
                rows.append(
                    {
                        "area_m2": area_m2,
                        "current_price": current_price,
                        "current_price_source": source,
                        "latest_price": info.latest_price,
                        "peak_price": info.peak_price,
                        "gap_ratio": gap_ratio,
                    }
                )
            return AgentResponse(
                success=True,
                message="\n".join(lines),
                data={"apt_name": resolved_name, "rows": rows},
                trace=trace,
            )

        info = _calc_peak_same_area(self.csv_path, resolved_name, target_area)
        if info is None:
            return AgentResponse(False, f"'{apt_name}'에 대한 거래 데이터를 찾지 못했습니다.", {}, trace)

        current_price, source = _current_price_from_naver_listings(
            self.naver_articles_path,
            resolved_name,
            info.area_m2,
        )
        if current_price is None:
            current_price = info.latest_price
            source = "실거래 최신가(폴백)"
        trace.append(f"current_price_source={source}")

        gap_ratio = (current_price / info.peak_price) * 100 if info.peak_price else 0.0
        msg = (
            f"{resolved_name} {info.area_m2}㎡형 전고점 비교\n"
            f"- 현재가(매물 기준): {current_price:,}만원\n"
            f"- 전고점: {info.peak_price:,}만원\n"
            f"- 전고점 대비: {gap_ratio:.1f}%"
        )
        return AgentResponse(
            success=True,
            message=msg,
            data={
                "apt_name": resolved_name,
                "area_m2": info.area_m2,
                "latest_price": info.latest_price,
                "current_price": current_price,
                "current_price_source": source,
                "peak_price": info.peak_price,
                "gap_ratio": gap_ratio,
            },
            trace=trace,
        )


def _extract_apt_name(text: str) -> str:
    cleaned = text
    for token in ["전고점", "현재가", "비교", "정보", "알려줘", "알려", "조회", "부탁해", "요", "?"]:
        cleaned = cleaned.replace(token, " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _wants_other_areas(text: str) -> bool:
    t = (text or "").lower().replace(" ", "")
    tokens = ["다른평형", "다른평형대", "다른평수", "다른면적", "평형별", "다른타입"]
    return any(tok in t for tok in tokens)


def _to_price(value: str) -> int:
    digits = "".join(ch for ch in (value or "") if ch.isdigit())
    return int(digits) if digits else 0


def _parse_korean_price_to_manwon(value: str) -> Optional[int]:
    text = (value or "").strip()
    if not text:
        return None
    # 예: "31억 5,000", "31억", "6억", "27억 8,000"
    m = re.search(r"(\d+)\s*억", text)
    eok = int(m.group(1)) if m else 0
    rest = 0
    if "억" in text:
        after = text.split("억", 1)[1]
        m2 = re.search(r"([\d,]+)", after)
        if m2:
            rest = int(m2.group(1).replace(",", ""))
    else:
        digits = re.sub(r"[^0-9]", "", text)
        if digits:
            return int(digits)
    total = eok * 10000 + rest
    return total if total > 0 else None


def _extract_target_area(text: str) -> Optional[int]:
    # 예: "84형", "84㎡", "전용 84", "34평"
    m_m2 = re.search(r"(\d{2,3})(?:\s*㎡|\s*m2|\s*형)", text.lower())
    if m_m2:
        return int(m_m2.group(1))
    m_py = re.search(r"(\d{2})(?:\s*평)", text.lower())
    if m_py:
        py = int(m_py.group(1))
        return int(round(py * 3.3058))
    return None


def _calc_peak_same_area(path: Path, apt_name: str, target_area: Optional[int]) -> Optional[PeakInfo]:
    area_prices: dict[int, list[int]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("단지명") or "").strip() != apt_name:
                continue
            area_raw = (row.get("전용면적(㎡)") or "").strip()
            try:
                area_key = int(round(float(area_raw)))
            except Exception:
                continue
            price = _to_price(row.get("거래금액(만원)", ""))
            if price > 0:
                area_prices.setdefault(area_key, []).append(price)

    if not area_prices:
        return None

    selected_area: Optional[int] = None
    if target_area is not None:
        # 가장 가까운 평형 선택 (예: 84 입력 시 84/85 중 매칭)
        selected_area = min(area_prices.keys(), key=lambda a: abs(a - target_area))
    else:
        # 면적 미지정이면 거래 건수가 가장 많은 평형을 기본 비교군으로 사용
        selected_area = max(area_prices.keys(), key=lambda a: len(area_prices[a]))

    prices = area_prices.get(selected_area, [])
    if not prices:
        return None
    peak = max(prices)
    latest = prices[-1]
    ratio = (latest / peak) * 100 if peak else 0.0
    return PeakInfo(latest_price=latest, peak_price=peak, gap_ratio=ratio, area_m2=selected_area)


def _calc_peak_multi_areas(path: Path, apt_name: str, top_n: int = 6) -> list[tuple[int, PeakInfo]]:
    area_prices: dict[int, list[int]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("단지명") or "").strip() != apt_name:
                continue
            area_raw = (row.get("전용면적(㎡)") or "").strip()
            try:
                area_key = int(round(float(area_raw)))
            except Exception:
                continue
            price = _to_price(row.get("거래금액(만원)", ""))
            if price > 0:
                area_prices.setdefault(area_key, []).append(price)

    if not area_prices:
        return []

    # 먼저 거래 건수로 상위 평형을 고른 뒤, 최종 출력은 면적 오름차순으로 정렬
    selected = sorted(area_prices.keys(), key=lambda a: len(area_prices[a]), reverse=True)[:top_n]
    order = sorted(selected)
    out: list[tuple[int, PeakInfo]] = []
    for area in order:
        prices = area_prices[area]
        if not prices:
            continue
        peak = max(prices)
        latest = prices[-1]
        ratio = (latest / peak) * 100 if peak else 0.0
        out.append((area, PeakInfo(latest_price=latest, peak_price=peak, gap_ratio=ratio, area_m2=area)))
    return out


def _resolve_apt_name(path: Path, query: str) -> Optional[str]:
    q = _normalize(query)
    if not q:
        return None

    counts: dict[str, int] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("단지명") or "").strip()
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1

    if query in counts:
        return query

    candidates: list[tuple[int, int, str]] = []
    for name, cnt in counts.items():
        n = _normalize(name)
        if not n:
            continue
        if n == q:
            candidates.append((3, cnt, name))
        elif n in q or q in n:
            candidates.append((2, cnt, name))
        elif any(tok and tok in n for tok in q.split()):
            candidates.append((1, cnt, name))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], len(x[2])), reverse=True)
    return candidates[0][2]


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣 ]", "", (text or "").lower()).strip()


def _current_price_from_naver_listings(
    naver_csv_path: Path,
    apt_name: str,
    area_m2: int,
) -> tuple[Optional[int], str]:
    if not naver_csv_path.exists():
        return None, "네이버 매물 파일 없음"

    rows = _read_naver_articles_rows(naver_csv_path)
    if not rows:
        return None, "네이버 매물 파싱 실패/빈 데이터"

    name_norm = _normalize(apt_name)
    candidates: list[tuple[int, int]] = []  # (abs_area_diff, price)
    for row in rows:
        raw_name = (row.get("articleName") or "").strip()
        if _normalize(raw_name) != name_norm:
            continue

        trade_type = (row.get("tradeType") or row.get("tradeTypeName") or "").strip()
        if trade_type != "매매":
            continue

        area_raw = (row.get("area1") or "").strip()
        try:
            area = int(round(float(area_raw)))
        except Exception:
            continue

        price = _parse_korean_price_to_manwon(row.get("dealOrWarrantPrc", ""))
        if price is None:
            continue

        candidates.append((abs(area - area_m2), price))

    if not candidates:
        return None, "네이버 매물 매칭 없음"

    # 같은/가까운 평형 우선, 대표 현재가는 중앙값 사용(이상치 완화)
    candidates.sort(key=lambda x: x[0])
    min_diff = candidates[0][0]
    filtered_prices = sorted([p for d, p in candidates if d == min_diff])
    mid = len(filtered_prices) // 2
    if len(filtered_prices) % 2 == 0:
        current = int(round((filtered_prices[mid - 1] + filtered_prices[mid]) / 2))
    else:
        current = filtered_prices[mid]
    return current, "네이버 매물(매매) 중앙값"


def _read_naver_articles_rows(path: Path) -> list[dict[str, str]]:
    # 첫 줄 주석(#네이버...)이 있는 파일 포맷 지원
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if "articleNo" in s and "articleName" in s:
            start_idx = i
            break
    parsed = "".join(lines[start_idx:])
    if not parsed.strip():
        return []

    reader = csv.DictReader(parsed.splitlines())
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows
