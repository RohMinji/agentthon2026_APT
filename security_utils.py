from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


PHONE_PATTERN = re.compile(r"(?<!\d)(01[016789]-?\d{3,4}-?\d{4})(?!\d)")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RRN_PATTERN = re.compile(r"(?<!\d)(\d{6}-?[1-4]\d{6})(?!\d)")
DETAIL_ADDR_PATTERN = re.compile(r"\b\d{1,4}\s?동\s?\d{1,4}\s?호\b")
BANK_ACCOUNT_PATTERN = re.compile(r"(?<!\d)(\d{2,4}-\d{2,6}-\d{2,6}|\d{10,16})(?!\d)")


def hash_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def hash_user_id(user_id: str) -> str:
    salt = os.getenv("AUDIT_SALT", "default_audit_salt")
    return hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()


def mask_pii(text: str) -> str:
    masked = text or ""
    masked = PHONE_PATTERN.sub("[PHONE_MASKED]", masked)
    masked = EMAIL_PATTERN.sub("[EMAIL_MASKED]", masked)
    masked = RRN_PATTERN.sub("[RRN_MASKED]", masked)
    masked = DETAIL_ADDR_PATTERN.sub("[ADDR_MASKED]", masked)
    masked = BANK_ACCOUNT_PATTERN.sub("[ACCOUNT_MASKED]", masked)
    return masked


def append_audit_log(
    *,
    request_id: str,
    user_id: str,
    selected_agent: str,
    input_text: str,
    output_text: str,
    log_path: Optional[Union[str, Path]] = None,
) -> None:
    path = Path(log_path or os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl"))
    payload = {
        "request_id": request_id,
        "user_id_hash": hash_user_id(user_id),
        "selected_agent": selected_agent,
        "input_hash": hash_text(input_text),
        "output_hash": hash_text(output_text),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
