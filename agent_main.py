import os
import asyncio
import threading
from collections import deque
from uuid import uuid4

import telebot
from apartment_agent import create_apartment_search_agent

import json
from openai import OpenAI

# ============================================
# 🔐 인증 방식 선택 (둘 중 하나만 True로 설정!)
# ============================================
# API Key 인증 - 아래 값을 본인 정보로 수정!
AZURE_OPENAI_API_KEY = "" # 👈 API Key 입력

credential = None  # API Key 사용 시 credential 불필요
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY

#  아래 값을 본인의 Azure OpenAI 정보로 수정하세요!
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4o" # "gpt-4.1"  # 또는 gpt-4.1

client = OpenAI(
    base_url="",
    api_key=""
)

FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "si_do": {"type": ["string", "null"]},
        "si_gungu": {"type": ["string", "null"]},
        "eupmyeondong": {"type": ["string", "null"]},
        "corridor_type": {"type": ["string", "null"], "enum": ["계단식", "복도식", None]},
        "elementary_yn": {"type": ["string", "null"], "enum": ["Y", "N", None]},
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
        "si_do","si_gungu","eupmyeondong","corridor_type","elementary_yn","heating_type",
        "min_households","max_households","min_parking_per_household","max_parking_per_household",
        "min_age","max_age","min_exclusive_area","max_exclusive_area","min_price_eok","max_price_eok"
    ],
    "additionalProperties": False
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

def extract_filters(user_query: str) -> dict:
    completion = client.chat.completions.create(
        model="gpt-4o",
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

TELEGRAM_TOKEN = ''
bot = telebot.TeleBot(TELEGRAM_TOKEN)

class Session:
    def __init__(self):
        self.agent = create_apartment_search_agent(csv_path="./data/apt_basic_info.csv")
        self.session_id = str(uuid4())       # tool 메모리 키
        self.user_history = deque(maxlen=6)  # 사용자 질문만 저장

sessions: dict[int, Session] = {}
sessions_lock = threading.Lock()

def get_session(user_id: int) -> Session:
    with sessions_lock:
        if user_id not in sessions:
            sessions[user_id] = Session()
        return sessions[user_id]


def drop_session(user_id: int) -> None:
    with sessions_lock:
        sessions.pop(user_id, None)


async def ask(user_id: int, user_input: str) -> str:
    s = get_session(user_id)

    history_text = "\n".join([f"- {q}" for q in s.user_history]) or "- (없음)"
    prompt = f"""
# Below is the history:
## {history_text}

# User Question:
## {user_input}
""".strip()
    
    #### 필터링 단계 추가 ####
    pre_result = json.dumps(extract_filters(prompt), ensure_ascii=False)
    ######################

    result = await s.agent.run(pre_result)  # thread 전달 안함 = LLM 히스토리 미저장
    s.user_history.append(pre_result)   # 사용자 질문만 저장

    print(f"\n[USER] {user_input}")
    print(f"[AGENT] {result.text}")

    return result.text


@bot.message_handler(commands=["new", "reset"])
def reset_session_cmd(message):
    user_id = message.from_user.id
    drop_session(user_id)
    bot.send_message(message.chat.id, "대화 컨텍스트를 초기화했습니다. 새로 시작합니다.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    user_input = (message.text or "").strip()

    if not user_input:
        bot.send_message(message.chat.id, "질문을 입력해주세요.")
        return

    try:
        bot.send_chat_action(message.chat.id, "typing")
        waiting_msg = bot.send_message(message.chat.id, "🤖 AI가 생각 중입니다...")

        ai_reply = asyncio.run(ask(user_id, user_input))

        bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=waiting_msg.message_id,
            text=ai_reply,
        )
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ 오류 발생: {str(e)}")


print("상태 표시 기능이 포함된 LLM 봇 가동 중...")

### main 함수 안에 아래 포함시키기
bot.infinity_polling()
