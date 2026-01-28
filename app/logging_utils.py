"""
Logging utilities.

Primary goals:
- Avoid leaking secrets (API keys, tokens) in logs.
- Reduce noisy third-party logs (e.g., httpx request lines).
"""

from __future__ import annotations

import logging
import re
from typing import Iterable


class RedactSecretsFilter(logging.Filter):
    """
    Best-effort redaction for secrets in log messages.

    This is intentionally conservative: it only redacts common patterns like
    query-string keys and obvious token prefixes.
    """

    _query_param_re = re.compile(r"(?i)\\b(apiKey|apikey|token|key|secret|password)=([^&\\s]+)")
    _json_kv_re = re.compile(
        r"(?i)(\"?(apiKey|token|key|secret|password)\"?\\s*[:=]\\s*)(\"?)[^\"\\s,}]+(\\3)"
    )
    _bearer_re = re.compile(r"(?i)\\bBearer\\s+([A-Za-z0-9._\\-]+)")
    _sk_re = re.compile(r"\\bsk-[A-Za-z0-9]{10,}\\b")

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 (Filter.filter)
        try:
            msg = record.getMessage()
        except Exception:
            return True

        redacted = msg
        redacted = self._query_param_re.sub(lambda m: f"{m.group(1)}=REDACTED", redacted)
        redacted = self._json_kv_re.sub(
            lambda m: f"{m.group(1)}{m.group(3)}REDACTED{m.group(4)}", redacted
        )
        redacted = self._sk_re.sub("sk-REDACTED", redacted)
        redacted = self._bearer_re.sub("Bearer REDACTED", redacted)

        if redacted != msg:
            # Replace the fully formatted message to avoid re-formatting with args.
            record.msg = redacted
            record.args = ()
        return True


_FILTER_NAME = "brooks_redact_secrets"


def _has_filter(filters: Iterable[logging.Filter], name: str) -> bool:
    return any(getattr(f, "name", None) == name for f in filters)


def install_log_safety() -> None:
    """
    Install log safety defaults:
    - Redact common secrets in log messages
    - Quiet noisy library loggers that can leak query params (httpx)
    """
    # Quiet noisy HTTP request line logs (can include apiKey/token in URL)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    redact_filter = RedactSecretsFilter()
    redact_filter.name = _FILTER_NAME  # type: ignore[attr-defined]

    # Attach to root logger and any existing handlers.
    root = logging.getLogger()
    if not _has_filter(root.filters, _FILTER_NAME):
        root.addFilter(redact_filter)
    for handler in root.handlers:
        if not _has_filter(handler.filters, _FILTER_NAME):
            handler.addFilter(redact_filter)

    # Best-effort: also attach to existing non-root handlers (e.g., uvicorn).
    for obj in logging.Logger.manager.loggerDict.values():
        if isinstance(obj, logging.Logger):
            for handler in obj.handlers:
                if not _has_filter(handler.filters, _FILTER_NAME):
                    handler.addFilter(redact_filter)
