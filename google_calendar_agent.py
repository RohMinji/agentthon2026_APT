from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import Field

try:
    from agent_framework import ai_function
except ImportError:
    def ai_function(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator


SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarTool:
    def __init__(
        self,
        timezone: str = "Asia/Seoul",
        calendar_id: str = "primary",
        secrets_dir: str | Path = "./.secrets",
    ):
        self.timezone = timezone
        self.calendar_id = calendar_id
        self.secrets_dir = Path(secrets_dir)
        self.credentials_path = self.secrets_dir / "credentials.json"
        self.token_path = self.secrets_dir / "token.json"

    @staticmethod
    def _missing_deps_message() -> str:
        return (
            "Google Calendar 의존성이 필요합니다. 아래 명령으로 설치하세요:\n"
            "pip install google-api-python-client google-auth google-auth-oauthlib"
        )

    @staticmethod
    def _parse_datetime(value: str, timezone: str) -> datetime:
        text = (value or "").strip()
        if len(text) < 16 or ":" not in text:
            raise ValueError(
                "날짜+시간이 필요합니다. 예: 2026-03-10 14:00 또는 2026-03-10T14:00:00+09:00"
            )

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                "시간 형식이 잘못되었습니다. 예: 2026-03-10 14:00 또는 2026-03-10T14:00:00+09:00"
            ) from exc

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(timezone))
        return dt

    @staticmethod
    def _event_datetime(event: dict, key: str, fallback_timezone: str) -> str | None:
        node = event.get(key, {}) if isinstance(event, dict) else {}
        if not isinstance(node, dict):
            return None
        if node.get("dateTime"):
            return node.get("dateTime")
        if node.get("date"):
            return f'{node.get("date")}T00:00:00+00:00'
        _ = fallback_timezone
        return None

    def _serialize_event(self, event: dict) -> dict:
        return {
            "event_id": event.get("id"),
            "summary": event.get("summary"),
            "description": event.get("description"),
            "location": event.get("location"),
            "htmlLink": event.get("htmlLink"),
            "status": event.get("status"),
            "start": self._event_datetime(event, "start", self.timezone),
            "end": self._event_datetime(event, "end", self.timezone),
        }

    def _as_json(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _auth_error(self, message: str) -> str:
        return self._as_json(
            {
                "status": "error",
                "error_type": "auth_error",
                "message": message,
            }
        )

    def _validation_error(self, message: str) -> str:
        return self._as_json(
            {
                "status": "error",
                "error_type": "validation_error",
                "message": message,
            }
        )

    def _runtime_error(self, exc: Exception) -> str:
        if exc.__class__.__name__ == "HttpError":
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status == 403:
                return self._as_json(
                    {
                        "status": "error",
                        "error_type": "permission_denied",
                        "message": (
                            "Google Calendar 권한이 부족합니다(HTTP 403). "
                            "Calendar API 활성화, OAuth 동의 화면 범위, 계정 캘린더 접근권을 확인하세요."
                        ),
                    }
                )
            if status == 401:
                return self._auth_error(
                    "인증이 만료되었거나 유효하지 않습니다(HTTP 401). token.json 삭제 후 다시 로그인하세요."
                )
        return self._as_json(
            {
                "status": "error",
                "error_type": "runtime_error",
                "message": str(exc),
            }
        )

    def _google_imports(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
        except ImportError as exc:
            raise ImportError(self._missing_deps_message()) from exc

        return Request, Credentials, InstalledAppFlow, build, HttpError

    def _get_service(self):
        Request, Credentials, InstalledAppFlow, build, _ = self._google_imports()

        self.secrets_dir.mkdir(parents=True, exist_ok=True)

        if not self.credentials_path.exists():
            raise FileNotFoundError(
                f"OAuth client 파일이 없습니다: {self.credentials_path}\n"
                "Google Cloud Console에서 OAuth Client ID(Desktop app)를 생성해 credentials.json을 저장하세요."
            )

        creds = None
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
                creds = flow.run_local_server(port=0)

            self.token_path.write_text(creds.to_json(), encoding="utf-8")

        return build("calendar", "v3", credentials=creds)

    def _list_events_raw(
        self,
        start_time: str,
        end_time: str,
        max_results: int = 50,
        query: str | None = None,
    ) -> list[dict]:
        service = self._get_service()
        start_dt = self._parse_datetime(start_time, self.timezone)
        end_dt = self._parse_datetime(end_time, self.timezone)

        if start_dt >= end_dt:
            raise ValueError("시작 시간이 종료 시간보다 빠르거나 같아야 합니다.")

        response = (
            service.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=start_dt.isoformat(),
                timeMax=end_dt.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=max_results,
                q=query,
            )
            .execute()
        )
        return response.get("items", [])

    def _find_conflicts(
        self,
        start_time: str,
        end_time: str,
        exclude_event_id: str | None = None,
    ) -> list[dict]:
        items = self._list_events_raw(start_time=start_time, end_time=end_time, max_results=250)
        conflicts = []
        for item in items:
            if exclude_event_id and item.get("id") == exclude_event_id:
                continue
            conflicts.append(self._serialize_event(item))
        return conflicts

    @ai_function(
        name="check_calendar_conflicts",
        description="지정한 시작/종료 시간 범위에 겹치는 기존 일정을 조회한다.",
    )
    def check_calendar_conflicts(
        self,
        start_time: Annotated[str, Field(description="시작 시간. 예: 2026-03-10 14:00")],
        end_time: Annotated[str, Field(description="종료 시간. 예: 2026-03-10 15:00")],
        exclude_event_id: Annotated[str | None, Field(description="수정 시 제외할 event_id")] = None,
    ) -> str:
        try:
            conflicts = self._find_conflicts(
                start_time=start_time,
                end_time=end_time,
                exclude_event_id=exclude_event_id,
            )
            return self._as_json(
                {
                    "status": "ok",
                    "calendar_id": self.calendar_id,
                    "timezone": self.timezone,
                    "start_time": start_time,
                    "end_time": end_time,
                    "conflict_count": len(conflicts),
                    "conflicts": conflicts,
                }
            )
        except ImportError as exc:
            return self._validation_error(str(exc))
        except FileNotFoundError as exc:
            return self._auth_error(str(exc))
        except ValueError as exc:
            return self._validation_error(str(exc))
        except Exception as exc:
            return self._runtime_error(exc)

    @ai_function(
        name="create_calendar_event",
        description="Google Calendar에 새 일정을 생성한다. 생성 전 충돌 확인 가능.",
    )
    def create_calendar_event(
        self,
        summary: Annotated[str, Field(description="일정 제목")],
        start_time: Annotated[str, Field(description="시작 시간. 예: 2026-03-10 14:00")],
        end_time: Annotated[str, Field(description="종료 시간. 예: 2026-03-10 15:00")],
        description: Annotated[str | None, Field(description="일정 설명")] = None,
        location: Annotated[str | None, Field(description="장소")] = None,
        check_conflicts: Annotated[
            bool, Field(description="생성 전 충돌 확인 여부. 기본 true")
        ] = True,
        allow_conflicts: Annotated[
            bool, Field(description="충돌이 있어도 생성할지 여부. 기본 false")
        ] = False,
    ) -> str:
        try:
            start_dt = self._parse_datetime(start_time, self.timezone)
            end_dt = self._parse_datetime(end_time, self.timezone)
            if start_dt >= end_dt:
                return self._validation_error("시작 시간이 종료 시간보다 빨라야 합니다.")

            conflicts = []
            if check_conflicts:
                conflicts = self._find_conflicts(start_time=start_time, end_time=end_time)
                if conflicts and not allow_conflicts:
                    return self._as_json(
                        {
                            "status": "conflict",
                            "message": "동일 시간대 기존 일정이 있습니다. allow_conflicts=true로 재시도하거나 시간을 변경하세요.",
                            "conflict_count": len(conflicts),
                            "conflicts": conflicts,
                        }
                    )

            service = self._get_service()
            body = {
                "summary": summary,
                "description": description,
                "location": location,
                "start": {"dateTime": start_dt.isoformat(), "timeZone": self.timezone},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": self.timezone},
            }
            event = (
                service.events()
                .insert(calendarId=self.calendar_id, body=body)
                .execute()
            )

            return self._as_json(
                {
                    "status": "created",
                    "calendar_id": self.calendar_id,
                    "event": self._serialize_event(event),
                    "conflict_checked": check_conflicts,
                    "conflict_count": len(conflicts),
                }
            )
        except ImportError as exc:
            return self._validation_error(str(exc))
        except FileNotFoundError as exc:
            return self._auth_error(str(exc))
        except ValueError as exc:
            return self._validation_error(str(exc))
        except Exception as exc:
            return self._runtime_error(exc)

    @ai_function(
        name="list_calendar_events",
        description="기간 내 Google Calendar 일정을 조회한다.",
    )
    def list_calendar_events(
        self,
        start_time: Annotated[str, Field(description="조회 시작 시간. 예: 2026-03-10 00:00")],
        end_time: Annotated[str, Field(description="조회 종료 시간. 예: 2026-03-11 00:00")],
        max_results: Annotated[int, Field(description="최대 반환 개수", ge=1, le=250)] = 20,
        query: Annotated[str | None, Field(description="일정 제목/본문 검색어")] = None,
    ) -> str:
        try:
            items = self._list_events_raw(
                start_time=start_time,
                end_time=end_time,
                max_results=max_results,
                query=query,
            )
            serialized = [self._serialize_event(x) for x in items]
            return self._as_json(
                {
                    "status": "ok",
                    "calendar_id": self.calendar_id,
                    "timezone": self.timezone,
                    "start_time": start_time,
                    "end_time": end_time,
                    "returned_count": len(serialized),
                    "events": serialized,
                }
            )
        except ImportError as exc:
            return self._validation_error(str(exc))
        except FileNotFoundError as exc:
            return self._auth_error(str(exc))
        except ValueError as exc:
            return self._validation_error(str(exc))
        except Exception as exc:
            return self._runtime_error(exc)

    @ai_function(
        name="update_calendar_event",
        description="기존 Google Calendar 일정을 수정한다. 수정 전 충돌 확인 가능.",
    )
    def update_calendar_event(
        self,
        event_id: Annotated[str, Field(description="수정할 일정 event_id")],
        summary: Annotated[str | None, Field(description="수정할 제목")] = None,
        start_time: Annotated[str | None, Field(description="수정할 시작 시간")] = None,
        end_time: Annotated[str | None, Field(description="수정할 종료 시간")] = None,
        description: Annotated[str | None, Field(description="수정할 설명")] = None,
        location: Annotated[str | None, Field(description="수정할 장소")] = None,
        check_conflicts: Annotated[
            bool, Field(description="수정 전 충돌 확인 여부. 기본 true")
        ] = True,
        allow_conflicts: Annotated[
            bool, Field(description="충돌이 있어도 수정할지 여부. 기본 false")
        ] = False,
    ) -> str:
        try:
            service = self._get_service()
            current = (
                service.events()
                .get(calendarId=self.calendar_id, eventId=event_id)
                .execute()
            )

            current_start = current.get("start", {}).get("dateTime")
            current_end = current.get("end", {}).get("dateTime")
            if not current_start or not current_end:
                return self._validation_error(
                    "종일(all-day) 일정 수정은 현재 지원하지 않습니다. dateTime 일정만 수정 가능합니다."
                )

            new_start = start_time or current_start
            new_end = end_time or current_end

            start_dt = self._parse_datetime(new_start, self.timezone)
            end_dt = self._parse_datetime(new_end, self.timezone)
            if start_dt >= end_dt:
                return self._validation_error("시작 시간이 종료 시간보다 빨라야 합니다.")

            conflicts = []
            if check_conflicts:
                conflicts = self._find_conflicts(
                    start_time=start_dt.isoformat(),
                    end_time=end_dt.isoformat(),
                    exclude_event_id=event_id,
                )
                if conflicts and not allow_conflicts:
                    return self._as_json(
                        {
                            "status": "conflict",
                            "message": "수정하려는 시간대에 기존 일정이 있습니다. allow_conflicts=true로 재시도하거나 시간을 변경하세요.",
                            "conflict_count": len(conflicts),
                            "conflicts": conflicts,
                        }
                    )

            body = {}
            if summary is not None:
                body["summary"] = summary
            if description is not None:
                body["description"] = description
            if location is not None:
                body["location"] = location
            body["start"] = {"dateTime": start_dt.isoformat(), "timeZone": self.timezone}
            body["end"] = {"dateTime": end_dt.isoformat(), "timeZone": self.timezone}

            updated = (
                service.events()
                .patch(calendarId=self.calendar_id, eventId=event_id, body=body)
                .execute()
            )

            return self._as_json(
                {
                    "status": "updated",
                    "calendar_id": self.calendar_id,
                    "event": self._serialize_event(updated),
                    "conflict_checked": check_conflicts,
                    "conflict_count": len(conflicts),
                }
            )
        except ImportError as exc:
            return self._validation_error(str(exc))
        except FileNotFoundError as exc:
            return self._auth_error(str(exc))
        except ValueError as exc:
            return self._validation_error(str(exc))
        except Exception as exc:
            return self._runtime_error(exc)

    @ai_function(
        name="delete_calendar_event",
        description="Google Calendar에서 event_id 기준으로 일정을 삭제한다.",
    )
    def delete_calendar_event(
        self,
        event_id: Annotated[str, Field(description="삭제할 일정 event_id")],
    ) -> str:
        try:
            service = self._get_service()
            service.events().delete(calendarId=self.calendar_id, eventId=event_id).execute()
            return self._as_json(
                {
                    "status": "deleted",
                    "calendar_id": self.calendar_id,
                    "event_id": event_id,
                }
            )
        except ImportError as exc:
            return self._validation_error(str(exc))
        except FileNotFoundError as exc:
            return self._auth_error(str(exc))
        except Exception as exc:
            return self._runtime_error(exc)


def create_google_calendar_agent(
    credential=None,
    timezone: str = "Asia/Seoul",
):
    from agent_framework.azure import AzureOpenAIChatClient
    from azure.identity import AzureCliCredential

    calendar_tool = GoogleCalendarTool(
        timezone=timezone,
        calendar_id="primary",
        secrets_dir=Path(__file__).resolve().parent / ".secrets",
    )

    return AzureOpenAIChatClient(
        credential=credential or AzureCliCredential()
    ).as_agent(
        name="GoogleCalendarAssistant",
        instructions=(
            "너는 Google Calendar 일정 관리 전문가다. "
            "일정 생성(create_calendar_event)과 수정(update_calendar_event) 전에 "
            "반드시 check_calendar_conflicts를 먼저 호출해 충돌을 확인하라. "
            "충돌이 있으면 사용자의 확인을 받기 전까지 생성/수정을 확정하지 마라. "
            "날짜만 있는 요청은 거절하고 날짜+시간 포맷 예시를 안내하라. "
            "최종 응답에는 event_id, htmlLink, 시작/종료 시각을 반드시 포함하라."
        ),
        tools=[
            calendar_tool.check_calendar_conflicts,
            calendar_tool.create_calendar_event,
            calendar_tool.list_calendar_events,
            calendar_tool.update_calendar_event,
            calendar_tool.delete_calendar_event,
        ],
    )
