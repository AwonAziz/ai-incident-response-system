"""Minimal HTTP control plane (stdlib only).

The Dockerfile exposes port 8080 and ``docker-compose.yml`` ships an anomaly
injector container, so the running pipeline needs a real API surface rather than
a decorative ``EXPOSE``:

======================================  ==========================================
``GET  /health``                        liveness + uptime
``GET  /stats``                         summary, incidents, dashboard, model state
``GET  /incidents``                     active incidents (``?status=``/``?limit=``)
``GET  /events``                        incident lifecycle feed
``GET  /metrics``                       per-cloud telemetry summary
``POST /incidents/<id>/ack``            acknowledge an incident
``POST /incidents/<id>/resolve``        resolve an incident
``POST /inject``                        push one or many metrics through the pipeline
======================================  ==========================================

It is intentionally not a web framework: the surface is small, read-mostly and
lives inside the same process as the pipeline.
"""

from __future__ import annotations

import hmac
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, ClassVar
from urllib.parse import parse_qs, urlparse

from src.core.enums import IncidentStatus
from src.ingestion.metric_schema import Metric

__all__ = ["ControlServer"]

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 1_000_000
MAX_DRAIN_BYTES = 16_000_000


class ControlServer:
    """Serves the control API in a background thread.

    Authentication is a bearer token (``API_TOKEN``). When it is unset the API
    runs open and says so once in the log - convenient for ``localhost`` and for
    the Docker healthcheck, unacceptable on a shared network. ``/health`` is
    always exempt so liveness checks do not need credentials.
    """

    #: routes that never require a token
    PUBLIC_ROUTES: ClassVar[frozenset[str]] = frozenset({"", "health"})

    def __init__(
        self,
        pipeline: Any,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        token: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.host = host
        self.port = int(port)
        self.token = (token or "").strip() or None
        self.rejected_requests = 0
        if self.token is None:
            logger.warning(
                "control API has no API_TOKEN set - it is open to anything that can reach %s:%d",
                host,
                port,
            )
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    # ── auth ────────────────────────────────────────────────────────────
    @property
    def auth_required(self) -> bool:
        return self.token is not None

    def authenticate(self, authorization: str | None) -> tuple[bool, str]:
        """Check an ``Authorization`` header. Returns ``(ok, reason)``."""
        if not self.auth_required:
            return True, "open"
        if not authorization:
            return False, "missing Authorization header"
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer":
            return False, "expected the Bearer scheme"
        if not hmac.compare_digest(token.strip(), self.token or ""):
            return False, "invalid token"
        return True, "ok"

    # ── lifecycle ──────────────────────────────────────────────────────
    def start(self) -> ControlServer:
        if self._httpd is not None:
            return self
        self._httpd = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self.port = self._httpd.server_address[1]
        self._thread = Thread(target=self._httpd.serve_forever, name="control-api", daemon=True)
        self._thread.start()
        logger.info("control API listening on http://%s:%d", self.host, self.port)
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}"

    def __enter__(self) -> ControlServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ── routing ────────────────────────────────────────────────────────
    def handle(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: bytes,
        authorization: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Pure routing function - testable without a socket."""
        pipeline = self.pipeline
        segments = [segment for segment in path.split("/") if segment]
        route = segments[0] if segments else ""

        if route not in self.PUBLIC_ROUTES:
            allowed, reason = self.authenticate(authorization)
            if not allowed:
                self.rejected_requests += 1
                logger.warning("rejected %s %s: %s", method, path, reason)
                return self._error(HTTPStatus.UNAUTHORIZED, reason)

        if route in self.PUBLIC_ROUTES:
            if method != "GET":
                return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "GET required")
            return HTTPStatus.OK, {
                "status": "ok" if pipeline.running else "stopping",
                "ticks": pipeline.ticks,
                "uptime_seconds": round((pipeline.summary()["uptime_seconds"]), 3),
                "active_incidents": pipeline.incident_manager.stats["active_count"],
                "auth": "required" if self.auth_required else "open",
                "rejected_requests": self.rejected_requests,
            }

        if segments[0] == "stats" and method == "GET":
            return HTTPStatus.OK, pipeline.stats()

        if segments[0] == "metrics" and method == "GET":
            return HTTPStatus.OK, pipeline.dashboard.state

        if segments[0] == "events" and method == "GET":
            limit = _int_arg(query, "limit", 25)
            return HTTPStatus.OK, {"events": [event.to_dict() for event in pipeline.incident_manager.events(limit)]}

        if segments[0] == "incidents":
            if method == "GET":
                limit = _int_arg(query, "limit", 25)
                status = query.get("status", [None])[0]
                manager = pipeline.incident_manager
                if status:
                    try:
                        wanted = IncidentStatus.parse(status)
                    except ValueError:
                        return self._error(HTTPStatus.BAD_REQUEST, f"unknown status {status!r}")
                    incidents = [item for item in manager.all() if item.status is wanted]
                else:
                    incidents = manager.active()
                return HTTPStatus.OK, {"incidents": [item.to_dict() for item in incidents[:limit]]}
            if method == "POST" and len(segments) == 3 and segments[2] in ("ack", "resolve"):
                incident_id = segments[1]
                try:
                    if segments[2] == "ack":
                        incident = pipeline.incident_manager.acknowledge(incident_id, "acknowledged via API")
                    else:
                        incident = pipeline.incident_manager.resolve(incident_id, "resolved via API")
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, f"unknown incident {incident_id}")
                except ValueError as exc:
                    return self._error(HTTPStatus.CONFLICT, str(exc))
                return HTTPStatus.OK, {"incident": incident.to_dict()}
            return self._error(HTTPStatus.METHOD_NOT_ALLOWED, "unsupported incidents route")

        if segments[0] == "inject" and method == "POST":
            payload, error = _parse_json(body)
            if error:
                return self._error(HTTPStatus.BAD_REQUEST, error)
            try:
                metrics, note = self._metrics_from_payload(payload)
            except (KeyError, TypeError, ValueError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            created = pipeline.inject(metrics)
            return HTTPStatus.OK, {
                "accepted": len(metrics),
                "note": note,
                "incidents": [incident.to_dict() for incident in created],
            }

        return self._error(HTTPStatus.NOT_FOUND, f"unknown route {path!r}")

    def _metrics_from_payload(self, payload: Any) -> tuple[list[Metric], str]:
        """Accept either raw metric documents or a compact sample description."""
        rows = payload if isinstance(payload, list) else payload.get("metrics")
        if isinstance(rows, list):
            metrics = []
            for row in rows:
                metric = Metric.from_dict(row)
                confidence = float(row.get("confidence", 0.0) or 0.0)
                if confidence:
                    metric = metric.with_detection(
                        anomaly_score=float(row.get("anomaly_score") or metric.value),
                        is_anomaly=bool(row.get("is_anomaly", True)),
                        confidence=confidence,
                    )
                metrics.append(metric)
            return metrics, "explicit metric documents"

        if not isinstance(payload, dict):
            raise TypeError("payload must be a JSON object or a list of metric documents")

        cloud = str(payload.get("cloud", "aws"))
        metric_name = str(payload["metric"])
        if "value" in payload:
            metric = self.pipeline.build_sample(
                cloud,
                metric_name,
                float(payload["value"]),
                resource_id=payload.get("resource_id"),
                service=payload.get("service"),
                region=payload.get("region"),
                confidence=float(payload.get("confidence", 1.0)),
            )
            return [metric], "single sample"

        count = int(payload.get("count", 1))
        collector = next(
            (item for item in self.pipeline.collectors if item.cloud.value == cloud.lower()),
            None,
        )
        if collector is None:
            raise KeyError(f"unknown cloud {cloud!r}")
        metrics = []
        for _ in range(max(1, min(count, 50))):
            produced = collector.collect_and_track()
            if metric_name != "*":
                produced = [item for item in produced if item.name == metric_name]
            metrics.extend(produced)
        return metrics, f"generated {len(metrics)} sample(s)"

    # ── http plumbing ──────────────────────────────────────────────────
    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "ai-incident-response/1.0"
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                server._respond(self, "GET")

            def do_POST(self) -> None:
                server._respond(self, "POST")

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("api %s - %s", self.address_string(), fmt % args)

        return Handler

    def _respond(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        parsed = urlparse(handler.path)
        length = int(handler.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._drain(handler, length)
            self._write(handler, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body too large"})
            return
        body = handler.rfile.read(length) if length else b""
        try:
            status, payload = self.handle(
                method,
                parsed.path,
                parse_qs(parsed.query),
                body,
                handler.headers.get("Authorization"),
            )
        except Exception as exc:
            logger.exception("control API error")
            status, payload = HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"}
        self._write(handler, status, payload)

    @staticmethod
    def _drain(handler: BaseHTTPRequestHandler, length: int) -> None:
        """Discard an oversized body before replying.

        Closing the socket while the client is still writing makes the client
        see a connection reset instead of the 413, so the bytes have to go.
        """
        remaining = min(length, MAX_DRAIN_BYTES)
        while remaining > 0:
            chunk = handler.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _write(self, handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, default=str).encode("utf-8")
        handler.send_response(int(status))
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(encoded)))
        if int(status) == HTTPStatus.UNAUTHORIZED.value:
            handler.send_header("WWW-Authenticate", 'Bearer realm="control-api"')
        handler.end_headers()
        handler.wfile.write(encoded)

    @staticmethod
    def _error(status: HTTPStatus, message: str) -> tuple[int, dict[str, Any]]:
        return HTTPStatus(status.value), {"error": message}


def _parse_json(body: bytes) -> tuple[Any, str | None]:
    if not body:
        return {}, None
    try:
        return json.loads(body.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON body: {exc}"


def _int_arg(query: dict[str, list[str]], name: str, default: int) -> int:
    raw = query.get(name, [None])[0]
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
