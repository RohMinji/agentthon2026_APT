from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from models import CreatedEvent


class CalendarClient:
    """Google Calendar wrapper with duplicate check by extendedProperties.private.unique_key."""

    def __init__(
        self,
        calendar_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        service_account_json: Optional[str] = None,
        auth_mode: str = "service_account",
    ):
        self.calendar_id = calendar_id or os.getenv("GOOGLE_CALENDAR_ID", "primary")
        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.service_account_json = service_account_json or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        self.auth_mode = auth_mode
        self._service = None

    def _get_service(self):
        if self._service is not None:
            return self._service

        try:
            from googleapiclient.discovery import build
        except Exception as exc:
            raise RuntimeError(
                "google-api-python-client 가 필요합니다. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            ) from exc

        scopes = ["https://www.googleapis.com/auth/calendar"]

        if self.auth_mode == "service_account":
            try:
                from google.oauth2.service_account import Credentials
            except Exception as exc:
                raise RuntimeError("google-auth 가 필요합니다. pip install google-auth") from exc

            creds = None
            if self.service_account_json:
                info = json.loads(self.service_account_json)
                creds = Credentials.from_service_account_info(info, scopes=scopes)
            elif self.credentials_path:
                path = Path(self.credentials_path)
                if not path.exists():
                    raise RuntimeError(f"GOOGLE_CREDENTIALS_PATH 파일을 찾을 수 없습니다: {path}")
                creds = Credentials.from_service_account_file(str(path), scopes=scopes)
            else:
                raise RuntimeError(
                    "서비스 계정 인증에 필요한 자격증명이 없습니다. GOOGLE_CREDENTIALS_PATH 또는 GOOGLE_SERVICE_ACCOUNT_JSON을 설정하세요."
                )

            self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
            return self._service

        if self.auth_mode == "oauth":
            try:
                from google_auth_oauthlib.flow import InstalledAppFlow
            except Exception as exc:
                raise RuntimeError(
                    "OAuth 인증에는 google-auth-oauthlib 가 필요합니다. pip install google-auth-oauthlib"
                ) from exc

            if not self.credentials_path:
                raise RuntimeError("OAuth 인증에는 GOOGLE_CREDENTIALS_PATH(client_secret json)가 필요합니다.")
            path = Path(self.credentials_path)
            if not path.exists():
                raise RuntimeError(f"GOOGLE_CREDENTIALS_PATH 파일을 찾을 수 없습니다: {path}")

            flow = InstalledAppFlow.from_client_secrets_file(str(path), scopes=scopes)
            creds = flow.run_local_server(port=0)
            self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
            return self._service

        raise RuntimeError(f"지원하지 않는 auth_mode 입니다: {self.auth_mode}")

    def _event_time_window(self, event: CreatedEvent) -> tuple[str, str]:
        time_min = (event.start - timedelta(days=1)).isoformat()
        time_max = (event.end + timedelta(days=1)).isoformat()
        return time_min, time_max

    def find_event_id_by_unique_key(self, unique_key: str, event: CreatedEvent) -> str | None:
        service = self._get_service()
        time_min, time_max = self._event_time_window(event)

        items = (
            service.events()
            .list(
                calendarId=self.calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                maxResults=250,
                orderBy="startTime",
            )
            .execute()
            .get("items", [])
        )

        for item in items:
            key = (
                item.get("extendedProperties", {})
                .get("private", {})
                .get("unique_key")
            )
            if key == unique_key:
                return item.get("id")
        return None

    def create_event(self, event: CreatedEvent) -> str:
        service = self._get_service()

        body = {
            "summary": event.title,
            "extendedProperties": {"private": {"unique_key": event.unique_key}},
        }

        if event.all_day:
            body["start"] = {"date": event.start.date().isoformat()}
            body["end"] = {"date": event.end.date().isoformat()}
        else:
            tzname = event.start.tzinfo.key if event.start.tzinfo and hasattr(event.start.tzinfo, "key") else "Asia/Seoul"
            body["start"] = {"dateTime": event.start.isoformat(), "timeZone": tzname}
            body["end"] = {"dateTime": event.end.isoformat(), "timeZone": tzname}

        created = service.events().insert(calendarId=self.calendar_id, body=body).execute()
        return created["id"]
