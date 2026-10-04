"""HTTP helper shared by the webhook-style notifiers.

Uses :mod:`urllib.request` so notification delivery adds no runtime dependency
to the project.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = ["post_json"]

logger = logging.getLogger(__name__)


class DeliveryError(RuntimeError):
    """Raised when a notification endpoint rejects or cannot be reached."""


def post_json(url: str, payload: dict[str, Any], timeout: float = 5.0, headers: dict[str, str] | None = None) -> str:
    """POST ``payload`` as JSON. Raises :class:`DeliveryError` on any failure."""
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "User-Agent": "ai-incident-response/1.0"}
    request_headers.update(headers or {})
    request = Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200] if exc.fp else ""
        raise DeliveryError(f"HTTP {exc.code} from {url}: {detail or exc.reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise DeliveryError(f"{type(exc).__name__} calling {url}: {exc}") from exc
    return f"HTTP {status}{f' - {text[:120]}' if text.strip() else ''}"
