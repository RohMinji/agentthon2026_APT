from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field

try:
    from agent_framework import ai_function
except ImportError:
    def ai_function(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


def load_apartment_dataframe(csv_path: str | Path = "./data/apt_basic_info_school.csv") -> pd.DataFrame:
    """Load and normalize apartment data into the query schema used by the agent tool."""
    columns = [
        "k-아파트명",
        "k-단지분류(아파트,주상복합등등)",
        "kapt도로명주소",
        "주소(시도)k-apt주소split",
        "주소(시군구)",
        "주소(읍면동)",
        "나머지주소",
        "주소(도로명)",
        "주소(도로상세주소)",
        "k-세대타입(분양형태)",
        "k-복도유형",
        "k-난방방식",
        "k-전체세대수",
        "k-건설사(시공사)",
        "k-전용면적별세대현황(60㎡이하)",
        "k-전용면적별세대현황(60㎡~85㎡이하)",
        "k-85㎡~135㎡이하",
        "k-135㎡초과",
        "주차대수",
        "단지승인일",
        "좌표X",
        "좌표Y",
        "초품아"
    ]

    df = pd.read_csv(csv_path)[columns] # , encoding="cp949"
    df.columns = [
        "아파트명",
        "단지분류",
        "도로명주소",
        "주소(시도)",
        "주소(시군구)",
        "주소(읍면동)",
        "나머지주소",
        "주소(도로명)",
        "주소(도로상세주소)",
        "분양형태",
        "복도유형",
        "난방방식",
        "세대수",
        "시공사",
        "전용60이하",
        "전용60_85",
        "전용85_135",
        "전용135초과",
        "주차대수",
        "단지승인일",
        "좌표X",
        "좌표Y",
        "초품아"
    ]

    df = df.drop(df[df["주차대수"] == 0].index, axis=0)
    df = df.drop(df[df["좌표X"].isnull()].index, axis=0)
    df = df.drop(df[~df["분양형태"].isin(["분양", "혼합"])].index, axis=0)
    df = df.drop(df[~df["복도유형"].isin(["계단식", "복도식", "혼합식"])].index, axis=0)
    df = df.drop(df[~df["난방방식"].isin(["개별난방", "지역난방", "중앙난방"])].index, axis=0)
    df = df.drop(df[~df['단지분류'].isin(['아파트','주상복합','중앙난방'])].index, axis=0)
    df = df.drop(df[df["단지승인일"].isnull()].index, axis=0).reset_index(drop=True)

    df["주차대수_세대당"] = (df["주차대수"] / df["세대수"]).round(1)

    df["단지승인일"] = pd.to_datetime(df["단지승인일"])
    current_year = datetime.now().year
    df["연차"] = (current_year - df["단지승인일"].dt.year).astype(int)

    df.loc[df["복도유형"] == "혼합식", "복도유형"] = "복도식"

    real_estate_path = Path(csv_path).with_name("real_estate.csv")
    real_estate_db = pd.read_csv(real_estate_path)

    real_estate_db['계약순서'] = real_estate_db['계약년월'].astype(int) * 100 + real_estate_db['계약일'].astype(int)
    real_estate_db_sorted = real_estate_db.sort_values(by='계약순서', ascending=False)

    real_estate_db = real_estate_db_sorted.drop_duplicates(
        subset=['도로명', '전용면적(㎡)'], keep='first'
    ).copy()
    real_estate_db['도로명'] = (
        real_estate_db['시군구'].str.split(' ').str[0]
        + ' '
        + real_estate_db['시군구'].str.split(' ').str[1]
        + ' '
        + real_estate_db['도로명']
    )

    real_estate_db['거래금액(만원)'] = real_estate_db['거래금액(만원)'].str.replace(',' ,'').astype(int)

    real_estate_db = real_estate_db[[
        '도로명', '단지명', '전용면적(㎡)', '거래금액(만원)', '건축년도'
    ]]
    real_estate_db.columns = ['도로명주소', '단지명', '전용면적', '거래금액(만원)', '건축년도']

    apt_db_F = pd.merge(real_estate_db, df, on='도로명주소', how='left')
    apt_db_F = apt_db_F.drop(apt_db_F[apt_db_F['아파트명'].isnull()].index, axis=0).reset_index(drop=True)
    apt_db_F["전용면적"] = pd.to_numeric(apt_db_F["전용면적"], errors="coerce")
    apt_db_F["거래금액(만원)"] = pd.to_numeric(apt_db_F["거래금액(만원)"], errors="coerce")
    apt_db_F = apt_db_F.dropna(subset=["전용면적", "거래금액(만원)"]).reset_index(drop=True)
    apt_db_F['연차'] = 2027 - apt_db_F['건축년도']

    return apt_db_F[
        [
            "아파트명",
            '단지명',
            "단지분류",
            "도로명주소",
            '전용면적',
            '거래금액(만원)',
            "주소(시도)",
            "주소(시군구)",
            "주소(읍면동)",
            "나머지주소",
            "주소(도로명)",
            "주소(도로상세주소)",
            "복도유형",
            "난방방식",
            "세대수",
            "좌표X",
            "좌표Y",
            "주차대수_세대당",
            "연차",
            "초품아"
        ]
    ].copy()


class ApartmentSearchTool:
    def __init__(self, apartment_df: pd.DataFrame):
        self.df = apartment_df.copy()
        self._filter_memory: dict[str, dict] = {}

    @staticmethod
    def _split_values(text: str | None) -> list[str]:
        if not text:
            return []
        separators = [",", "/", "|", " 또는 ", " 혹은 ", " and "]
        normalized = text
        for sep in separators:
            normalized = normalized.replace(sep, ",")
        return [x.strip() for x in normalized.split(",") if x.strip()]

    def _intersect_csv_values(self, old_value: str | None, new_value: str | None) -> str | None:
        old_list = self._split_values(old_value)
        new_list = self._split_values(new_value)
        if not old_list:
            return new_value
        if not new_list:
            return old_value
        new_set = set(new_list)
        intersection = [v for v in old_list if v in new_set]
        if not intersection:
            return "__NO_MATCH__"
        return ",".join(intersection)

    @staticmethod
    def _expand_clear_filter_aliases(fields: list[str]) -> list[str]:
        alias_map = {
            "가격": ["min_price_eok", "max_price_eok", "min_price_10k_krw", "max_price_10k_krw"],
            "금액": ["min_price_eok", "max_price_eok", "min_price_10k_krw", "max_price_10k_krw"],
            "price": ["min_price_eok", "max_price_eok", "min_price_10k_krw", "max_price_10k_krw"],
            "price_filter": ["min_price_eok", "max_price_eok", "min_price_10k_krw", "max_price_10k_krw"],
            "면적": ["min_exclusive_area", "max_exclusive_area", "target_exclusive_area", "exclusive_area_mode", "area_tolerance"],
            "전용면적": ["min_exclusive_area", "max_exclusive_area", "target_exclusive_area", "exclusive_area_mode", "area_tolerance"],
            "area": ["min_exclusive_area", "max_exclusive_area", "target_exclusive_area", "exclusive_area_mode", "area_tolerance"],
            "연식": ["min_age", "max_age"],
            "연차": ["min_age", "max_age"],
            "age": ["min_age", "max_age"],
            "세대수": ["min_households", "max_households"],
            "households": ["min_households", "max_households"],
            "위치": ["si_do", "si_gungu", "eupmyeondong"],
            "지역": ["si_do", "si_gungu", "eupmyeondong"],
            "location": ["si_do", "si_gungu", "eupmyeondong"],
            "난방": ["heating_type"],
            "heating": ["heating_type"],
            "복도": ["corridor_type"],
            "corridor": ["corridor_type"],
            "초품아":  ["elementary_yn"]
        }
        valid_filter_keys = {
            "si_do",
            "si_gungu",
            "eupmyeondong",
            "corridor_type",
            "elementary_yn",
            "heating_type",
            "min_households",
            "max_households",
            "min_parking_per_household",
            "max_parking_per_household",
            "min_age",
            "max_age",
            "min_exclusive_area",
            "max_exclusive_area",
            "target_exclusive_area",
            "exclusive_area_mode",
            "area_tolerance",
            "min_price_10k_krw",
            "max_price_10k_krw",
            "min_price_eok",
            "max_price_eok",
        }
        expanded: list[str] = []
        for field in fields:
            normalized = field.strip().lower().replace(" ", "_")
            if normalized in alias_map:
                expanded.extend(alias_map[normalized])
                continue

            matched = False
            for alias, mapped_keys in alias_map.items():
                if alias and alias in normalized:
                    expanded.extend(mapped_keys)
                    matched = True
            if matched:
                continue

            if field.strip() in valid_filter_keys:
                expanded.append(field.strip())
        # Deduplicate while keeping order.
        return list(dict.fromkeys([x for x in expanded if x]))

    def _merge_filters_append(self, previous: dict, current: dict) -> dict:
        merged = dict(previous)
        for key, value in current.items():
            if value is None:
                continue

            if key in {"si_do", "si_gungu", "eupmyeondong"}:
                merged[key] = self._intersect_csv_values(merged.get(key), value)
                continue

            if key in {"corridor_type", "heating_type", "elementary_yn"}:
                if merged.get(key) is None:
                    merged[key] = value
                elif merged.get(key) != value:
                    merged[key] = "__NO_MATCH__"
                continue

            if key in {"min_households", "min_parking_per_household", "min_age", "min_exclusive_area"}:
                if merged.get(key) is None:
                    merged[key] = value
                else:
                    merged[key] = max(merged[key], value)
                continue

            if key in {"max_households", "max_parking_per_household", "max_age", "max_exclusive_area"}:
                if merged.get(key) is None:
                    merged[key] = value
                else:
                    merged[key] = min(merged[key], value)
                continue

            if key in {"min_price_10k_krw", "min_price_eok"}:
                if merged.get(key) is None:
                    merged[key] = value
                else:
                    merged[key] = max(merged[key], value)
                continue

            if key in {"max_price_10k_krw", "max_price_eok"}:
                if merged.get(key) is None:
                    merged[key] = value
                else:
                    merged[key] = min(merged[key], value)
                continue

            if key == "area_tolerance":
                if merged.get(key) is None:
                    merged[key] = value
                else:
                    merged[key] = min(float(merged[key]), float(value))
                continue

            if key == "exclusive_area_mode":
                # Keep prior mode in append to avoid accidental widening.
                merged.setdefault(key, value)
                continue

            # target_exclusive_area, limit and others: use latest value.
            merged[key] = value

        return merged

    @ai_function(
        name="search_apartments",
        description="사용자 조건(위치, 세대수, 전용면적, 가격 등)에 맞는 아파트 실거래 매물을 검색한다.",
    )
    def search_apartments(
        self,
        si_do: Annotated[
            str | None,
            Field(description="주소(시도). 여러 값이면 콤마(,)로 구분. 예: 서울,경기"),
        ] = None,
        si_gungu: Annotated[
            str | None,
            Field(description="주소(시군구). 여러 값이면 콤마(,)로 구분. 예: 송파구,서초구"),
        ] = None,
        eupmyeondong: Annotated[
            str | None,
            Field(description="주소(읍면동). 여러 값이면 콤마(,)로 구분."),
        ] = None,
        corridor_type: Annotated[
            str | None,
            Field(description="복도유형. 허용값: 계단식, 복도식"),
        ] = None,
        elementary_yn: Annotated[
            str | None,
            Field(description="초품아(초등학교가 인근에 위치하였는지 여부). 허용값: Y, N"),
        ] = None,
        heating_type: Annotated[
            str | None,
            Field(description="난방방식. 허용값: 개별난방, 지역난방, 중앙난방"),
        ] = None,
        min_households: Annotated[int | None, Field(description="세대수 최소값")] = None,
        max_households: Annotated[int | None, Field(description="세대수 최대값")] = None,
        min_parking_per_household: Annotated[
            float | None, Field(description="주차대수_세대당 최소값")
        ] = None,
        max_parking_per_household: Annotated[
            float | None, Field(description="주차대수_세대당 최대값")
        ] = None,
        min_age: Annotated[int | None, Field(description="연차 최소값(년)")] = None,
        max_age: Annotated[int | None, Field(description="연차 최대값(년)")] = None,
        min_exclusive_area: Annotated[
            float | None, Field(description="전용면적 최소값(㎡)")
        ] = None,
        max_exclusive_area: Annotated[
            float | None, Field(description="전용면적 최대값(㎡)")
        ] = None,
        target_exclusive_area: Annotated[
            float | None, Field(description="원하는 전용면적(㎡). 예: 84")
        ] = None,
        exclusive_area_mode: Annotated[
            str, Field(description="target_exclusive_area 적용 방식: exact/gte/lte")
        ] = "gte",
        area_tolerance: Annotated[
            float, Field(description="target_exclusive_area 허용 오차(㎡)", ge=0.0, le=10.0)
        ] = 0.5,
        min_price_10k_krw: Annotated[
            int | None, Field(description="거래금액(만원) 최소값")
        ] = None,
        max_price_10k_krw: Annotated[
            int | None, Field(description="거래금액(만원) 최대값")
        ] = None,
        min_price_eok: Annotated[
            float | None, Field(description="거래금액 최소값(억원). 예: 10")
        ] = None,
        max_price_eok: Annotated[
            float | None, Field(description="거래금액 최대값(억원). 예: 20")
        ] = None,
        limit: Annotated[int, Field(description="최대 반환 행 수", ge=1, le=300)] = 100,
        thread_id: Annotated[
            str | None, Field(description="대화 스레드 식별자. 같은 값이면 조건을 이어서 사용 가능")
        ] = None,
        memory_mode: Annotated[
            str, Field(description="필터 메모리 처리 방식: append/replace")
        ] = "append",
        reset_memory: Annotated[
            bool, Field(description="현재 thread_id의 누적 필터를 초기화할지 여부")
        ] = False,
        confirm_reset: Annotated[
            bool, Field(description="reset_memory 적용 확인 여부. true일 때만 초기화")
        ] = False,
        clear_filters: Annotated[
            str | None,
            Field(
                description=(
                    "누적 필터에서 제거할 필드명/조건명/문구 목록(콤마 구분). "
                    "예: max_price_eok,min_exclusive_area"
                    "예: 연차,난방 / '면적 조건 빼줘'"
                )
            ),
        ] = None
    ) -> str:
        """아파트 조건 검색 도구. 전달된 조건으로 데이터프레임을 필터링해 결과를 반환한다."""
        memory_key = "_default"
        if thread_id and (thread_id in self._filter_memory):
            memory_key = thread_id
        mode = (memory_mode or "append").lower()
        if mode not in {"append", "replace"}:
            mode = "append"
        # Prevent accidental replacement in normal multi-turn usage.
        # Start-new-search should happen via reset_memory/reset tool first.
        if mode == "replace" and not reset_memory:
            mode = "append"

        if reset_memory and confirm_reset:
            self._filter_memory.pop(memory_key, None)

        current_filters = {
            "si_do": si_do,
            "si_gungu": si_gungu,
            "eupmyeondong": eupmyeondong,
            "corridor_type": corridor_type,
            "elementary_yn": elementary_yn,
            "heating_type": heating_type,
            "min_households": min_households,
            "max_households": max_households,
            "min_parking_per_household": min_parking_per_household,
            "max_parking_per_household": max_parking_per_household,
            "min_age": min_age,
            "max_age": max_age,
            "min_exclusive_area": min_exclusive_area,
            "max_exclusive_area": max_exclusive_area,
            "target_exclusive_area": target_exclusive_area,
            "exclusive_area_mode": exclusive_area_mode,
            "area_tolerance": area_tolerance,
            "min_price_10k_krw": min_price_10k_krw,
            "max_price_10k_krw": max_price_10k_krw,
            "min_price_eok": min_price_eok,
            "max_price_eok": max_price_eok,
            "limit": limit,
        }
        specified = {k: v for k, v in current_filters.items() if v is not None}

        if mode == "append":
            merged_filters = self._merge_filters_append(
                self._filter_memory.get(memory_key, {}),
                specified,
            )
        else:
            merged_filters = dict(specified)

        fields_to_clear = self._split_values(clear_filters)
        fields_to_clear = self._expand_clear_filter_aliases(fields_to_clear)
        for field_name in fields_to_clear:
            merged_filters.pop(field_name, None)

        merged_filters.setdefault("exclusive_area_mode", "gte")
        merged_filters.setdefault("area_tolerance", 0.5)
        merged_filters.setdefault("limit", 100)
        self._filter_memory[memory_key] = dict(merged_filters)

        df = self.df.copy()
        exclusive_area_mode = str(merged_filters.get("exclusive_area_mode", "gte")).lower()
        min_price_candidates = []
        max_price_candidates = []
        if merged_filters.get("min_price_10k_krw") is not None:
            v = int(merged_filters["min_price_10k_krw"])
            if 0 < v <= 500:
                v *= 10000
            min_price_candidates.append(v)
        if merged_filters.get("min_price_eok") is not None:
            min_price_candidates.append(int(float(merged_filters["min_price_eok"]) * 10000))
        if merged_filters.get("max_price_10k_krw") is not None:
            v = int(merged_filters["max_price_10k_krw"])
            if 0 < v <= 500:
                v *= 10000
            max_price_candidates.append(v)
        if merged_filters.get("max_price_eok") is not None:
            max_price_candidates.append(int(float(merged_filters["max_price_eok"]) * 10000))

        min_price_10k_krw = max(min_price_candidates) if min_price_candidates else None
        max_price_10k_krw = min(max_price_candidates) if max_price_candidates else None

        values = self._split_values(merged_filters.get("si_do"))
        if values:
            df = df[df["주소(시도)"].isin(values)]

        values = self._split_values(merged_filters.get("si_gungu"))
        if values:
            df = df[df["주소(시군구)"].isin(values)]

        values = self._split_values(merged_filters.get("eupmyeondong"))
        if values:
            df = df[df["주소(읍면동)"].isin(values)]

        corridor_type = merged_filters.get("corridor_type")
        if corridor_type:
            df = df[df["복도유형"] == corridor_type]

        elementary_yn = merged_filters.get("elementary_yn")
        if elementary_yn:
            df = df[df["초품아"] == elementary_yn]

        heating_type = merged_filters.get("heating_type")
        if heating_type:
            df = df[df["난방방식"] == heating_type]

        min_households = merged_filters.get("min_households")
        max_households = merged_filters.get("max_households")
        if min_households is not None:
            df = df[df["세대수"] >= min_households]
        if max_households is not None:
            df = df[df["세대수"] <= max_households]

        min_parking_per_household = merged_filters.get("min_parking_per_household")
        max_parking_per_household = merged_filters.get("max_parking_per_household")
        if min_parking_per_household is not None:
            df = df[df["주차대수_세대당"] >= min_parking_per_household]
        if max_parking_per_household is not None:
            df = df[df["주차대수_세대당"] <= max_parking_per_household]

        min_age = merged_filters.get("min_age")
        max_age = merged_filters.get("max_age")
        if min_age is not None:
            df = df[df["연차"] >= min_age]
        if max_age is not None:
            df = df[df["연차"] <= max_age]

        min_exclusive_area = merged_filters.get("min_exclusive_area")
        max_exclusive_area = merged_filters.get("max_exclusive_area")
        target_exclusive_area = merged_filters.get("target_exclusive_area")
        area_tolerance = float(merged_filters.get("area_tolerance", 0.5))
        if min_exclusive_area is not None:
            df = df[df["전용면적"] >= min_exclusive_area]
        if max_exclusive_area is not None:
            df = df[df["전용면적"] <= max_exclusive_area]
        if target_exclusive_area is not None:
            if exclusive_area_mode == "gte":
                df = df[df["전용면적"] >= target_exclusive_area]
            elif exclusive_area_mode == "lte":
                df = df[df["전용면적"] <= target_exclusive_area]
            else:
                df = df[
                    (df["전용면적"] >= target_exclusive_area - area_tolerance)
                    & (df["전용면적"] <= target_exclusive_area + area_tolerance)
                ]

        if min_price_10k_krw is not None:
            df = df[df["거래금액(만원)"] >= min_price_10k_krw]
        if max_price_10k_krw is not None:
            df = df[df["거래금액(만원)"] <= max_price_10k_krw]

        limit = int(merged_filters.get("limit", 100))
        if df.empty:
            payload = {
                "total_count": 0,
                "returned_count": 0,
                "limit": 0,
                "thread_id": memory_key,
                "memory_mode": mode,
                "applied_filters": merged_filters,
                "rows": [],
            }
            return pd.Series(payload).to_json(force_ascii=False)

        show_cols = [
            "아파트명",
            # "주소(시도)",
            "주소(시군구)",
            "주소(읍면동)",
            "전용면적",
            "세대수",
            "거래금액(만원)",
            "초품아",
            "연차",
            "복도유형",
            "난방방식",
            "주차대수_세대당",
        ]
        sorted_df = df.sort_values(["거래금액(만원)", "연차"], ascending=[True, True])
        result = sorted_df[show_cols].head(limit)
        payload = {
            "total_count": int(len(sorted_df)),
            "returned_count": int(len(result)),
            "limit": int(limit),
            "thread_id": memory_key,
            "memory_mode": mode,
            "applied_filters": merged_filters,
            "effective_price_min_10k_krw": min_price_10k_krw,
            "effective_price_max_10k_krw": max_price_10k_krw,
            "rows": result.to_dict(orient="records"),
        }
        return pd.Series(payload).to_json(force_ascii=False)

    @ai_function(
        name="reset_apartment_search_memory",
        description="특정 thread_id의 누적 검색 조건을 초기화한다.",
    )
    def reset_apartment_search_memory(
        self,
        thread_id: Annotated[
            str | None, Field(description="초기화할 대화 스레드 식별자. 미입력 시 기본값")
        ] = None,
        confirm_reset: Annotated[
            bool, Field(description="초기화 확인 여부. true일 때만 실제 초기화")
        ] = False,
    ) -> str:
        memory_key = thread_id or "_default"
        if not confirm_reset:
            return pd.Series(
                {
                    "thread_id": memory_key,
                    "status": "skipped",
                    "reason": "confirm_reset=true required",
                }
            ).to_json(force_ascii=False)
        self._filter_memory.pop(memory_key, None)
        return pd.Series({"thread_id": memory_key, "status": "reset"}).to_json(force_ascii=False)


def create_apartment_search_agent(
    csv_path: str | Path = "./data/apt_basic_info_school.csv",
    credential=None,
):
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity import AzureCliCredential

    apartment_df = load_apartment_dataframe(csv_path)
    search_tool = ApartmentSearchTool(apartment_df)

    return AzureOpenAIChatClient(
        credential=credential or AzureCliCredential()
    ).as_agent(
        name="ApartmentFinder",
        instructions=(
            "너는 아파트 검색 전문가다. 사용자의 자연어 요청에서 조건을 추출해 search_apartments 도구를 호출하라. "
            "결과가 JSON으로 반환되면 반드시 total_count(전체 건수)와 returned_count(표시 건수)를 먼저 명시하고 "
            "'상위 N건' 형태로 표를 보여줘라. "
            "멀티턴 대화에서는 반드시 같은 thread_id를 유지해서 호출해라. "
            # "이전 조건에 추가 조건이면 memory_mode='append'를 사용해라. "
            "사용자가 '초기화'를 요청하면 reset_apartment_search_memory를 호출해라. "
            "사용자가 특정 조건만 빼거나 제외 해달라고 하면 memory_mode='append'에서 clear_filters를 사용해라. "
            "예: '금액 필터 빼줘', '연차 조건 제외', '난방 조건 제거'"
            "search_apartments 호출 시에는 reset_memory를 사용하지 말아라(필요시 confirm_reset=true가 있어야 동작). "
            "주의: 일반 대화에서는 memory_mode='replace'를 사용하지 말고 append를 유지해라. "
            "사용자가 전용면적을 말하면 target_exclusive_area 또는 면적 범위(min_exclusive_area/max_exclusive_area)로 매핑하라. "
            "예: 전용 84는 target_exclusive_area=84. '84 이상'은 min_exclusive_area=84 또는 "
            "target_exclusive_area=84, exclusive_area_mode='gte'로 호출하라. "
            "exclusive_area_mode를 생략하면 기본값은 'gte'다. exact는 '정확히/근처/84형만'처럼 명시된 경우에만 사용하라. "
            "가격 조건은 거래금액(만원) 또는 억원(min_price_eok/max_price_eok)으로 넣어라. "
            "예: 20억 미만은 max_price_eok=20. "
            "20억과 2억을 혼동하지 말고, 억 단위는 항상 정확한 숫자로 다시 확인해서 설명해라."
            "사용자가 사용하지 않은 필터 중에 추가 하면 좋다고 판단되는 필터는 역으로 사용할 것을 제안해라. 예를 들어, 500 세대 이상으로 찾아보는 건 어떨까요?"
        ),
        tools=[search_tool.search_apartments, search_tool.reset_apartment_search_memory],
    )
