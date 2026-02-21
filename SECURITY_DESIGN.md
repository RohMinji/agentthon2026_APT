# Security Design Summary

## 1) Capability-based control
- 모든 서브에이전트는 `capabilities`를 선언한다.
- 오케스트레이터(`orchestrator_core.py`)가 실행 전 누락 권한을 검사하고 차단한다.
- 차단 시 `errors`에 `capability_blocked:*`를 기록한다.

## 2) PII masking
- `security_utils.py::mask_pii`에서 다음 패턴을 마스킹한다.
- 전화번호, 이메일, 주민번호 패턴, 상세주소 동/호, 계좌번호
- QA 리포트 에이전트는 원문을 직접 노출하지 않고 마스킹 결과를 사용한다.

## 3) Audit logging (JSONL)
- 기록 필드:
  - `request_id`
  - `user_id_hash` (salted hash)
  - `selected_agent`
  - `input_hash`
  - `output_hash`
  - `timestamp` (UTC)
- 저장 경로는 `AUDIT_LOG_PATH` 환경변수로 제어한다.

## 4) Secret management
- API/SMTP 토큰은 코드 하드코딩 금지
- `.env` 기반 로딩
- 샘플은 `.env.example` 제공

## 5) FAQ anti-hallucination
- PDF citation 없으면 답변 거절
- 문서 근거가 없고 외부 검증이 필요하면 `web_search` capability 하에서 3개 이상 독립 출처 검증 필요
- 프롬프트 인젝션 유사 문구는 문서 전처리 단계에서 제거
