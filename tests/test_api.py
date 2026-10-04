"""Control API: pure routing logic plus a real HTTP round trip."""

from __future__ import annotations

import dataclasses
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from src.api.control_server import ControlServer
from src.core.clock import ManualClock
from src.pipeline import Pipeline, PipelineConfig


@pytest.fixture
def api_pipeline(settings) -> Pipeline:
    """Small training set: this module exercises HTTP, not model quality."""
    config = PipelineConfig(
        settings=dataclasses.replace(settings, baseline_samples=12, model_n_estimators=10),
        retrain=True,
        dashboard=False,
        console_notifications=False,
    )
    return Pipeline(config, clock=ManualClock()).prepare()


@pytest.fixture
def server(api_pipeline: Pipeline):
    instance = ControlServer(api_pipeline, host="127.0.0.1", port=0).start()
    yield instance
    instance.stop()


def _call(
    server: ControlServer,
    method: str,
    path: str,
    body: dict | None = None,
    token: str | None = None,
) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"{server.url}{path}",
        data=data,
        method=method,
        headers=headers,
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


class TestRoutingLogic:
    """``handle`` is a pure function, so most of the surface needs no socket."""

    def test_health(self, api_pipeline: Pipeline) -> None:
        server = ControlServer(api_pipeline, port=0)
        status, payload = server.handle("GET", "/health", {}, b"")
        assert status == 200
        assert payload["status"] == "ok"
        assert payload["active_incidents"] == 0

    def test_stats(self, api_pipeline: Pipeline) -> None:
        status, payload = ControlServer(api_pipeline, port=0).handle("GET", "/stats", {}, b"")
        assert status == 200
        assert "summary" in payload and "incidents" in payload

    def test_metrics(self, api_pipeline: Pipeline) -> None:
        api_pipeline.tick()
        status, payload = ControlServer(api_pipeline, port=0).handle("GET", "/metrics", {}, b"")
        assert status == 200
        assert payload["metrics_seen"] == 55

    def test_events(self, api_pipeline: Pipeline) -> None:
        status, payload = ControlServer(api_pipeline, port=0).handle("GET", "/events", {}, b"")
        assert status == 200
        assert payload["events"] == []

    def test_unknown_route(self, api_pipeline: Pipeline) -> None:
        status, payload = ControlServer(api_pipeline, port=0).handle("GET", "/nope", {}, b"")
        assert status == 404
        assert "unknown route" in payload["error"]

    def test_method_not_allowed(self, api_pipeline: Pipeline) -> None:
        status, _ = ControlServer(api_pipeline, port=0).handle("POST", "/health", {}, b"")
        assert status == 405

    def test_invalid_json(self, api_pipeline: Pipeline) -> None:
        status, payload = ControlServer(api_pipeline, port=0).handle("POST", "/inject", {}, b"{not json")
        assert status == 400
        assert "invalid JSON" in payload["error"]

    def test_incidents_status_filter(self, api_pipeline: Pipeline) -> None:
        created = api_pipeline.inject_sample("aws", "cpu_utilization", 99.0)
        server = ControlServer(api_pipeline, port=0)
        status, payload = server.handle("GET", "/incidents", {"status": ["open"]}, b"")
        assert status == 200
        assert created[0].id in {item["id"] for item in payload["incidents"]}
        status, payload = server.handle("GET", "/incidents", {"status": ["resolved"]}, b"")
        assert payload["incidents"] == []
        status, payload = server.handle("GET", "/incidents", {"status": ["bogus"]}, b"")
        assert status == 400

    def test_ack_and_resolve(self, api_pipeline: Pipeline) -> None:
        incident = api_pipeline.inject_sample("aws", "cpu_utilization", 99.0)[0]
        server = ControlServer(api_pipeline, port=0)
        status, payload = server.handle("POST", f"/incidents/{incident.id}/ack", {}, b"")
        assert status == 200
        assert payload["incident"]["status"] == "acknowledged"
        status, payload = server.handle("POST", f"/incidents/{incident.id}/resolve", {}, b"")
        assert payload["incident"]["status"] == "resolved"

    def test_ack_unknown_incident(self, api_pipeline: Pipeline) -> None:
        status, _ = ControlServer(api_pipeline, port=0).handle("POST", "/incidents/INC-999/ack", {}, b"")
        assert status == 404

    def test_ack_resolved_incident_conflicts(self, api_pipeline: Pipeline) -> None:
        incident = api_pipeline.inject_sample("aws", "cpu_utilization", 99.0)[0]
        server = ControlServer(api_pipeline, port=0)
        server.handle("POST", f"/incidents/{incident.id}/resolve", {}, b"")
        status, _ = server.handle("POST", f"/incidents/{incident.id}/ack", {}, b"")
        assert status == 409

    def test_unsupported_incidents_route(self, api_pipeline: Pipeline) -> None:
        assert ControlServer(api_pipeline, port=0).handle("DELETE", "/incidents", {}, b"")[0] == 405

    def test_inject_sample_payload(self, api_pipeline: Pipeline) -> None:
        server = ControlServer(api_pipeline, port=0)
        status, payload = server.handle(
            "POST",
            "/inject",
            {},
            json.dumps({"cloud": "gcp", "metric": "query_latency_ms", "value": 300.0}).encode(),
        )
        assert status == 200
        assert payload["accepted"] == 1
        assert payload["incidents"][0]["cloud"] == "gcp"

    def test_inject_metric_documents(self, api_pipeline: Pipeline) -> None:
        rows = api_pipeline.build_sample("aws", "cpu_utilization", 99.0).to_dict()
        rows["confidence"] = 0.95
        server = ControlServer(api_pipeline, port=0)
        status, payload = server.handle("POST", "/inject", {}, json.dumps({"metrics": [rows]}).encode())
        assert status == 200
        assert payload["incidents"][0]["metric_name"] == "cpu_utilization"

    def test_inject_generated_samples(self, api_pipeline: Pipeline) -> None:
        server = ControlServer(api_pipeline, port=0)
        status, payload = server.handle(
            "POST",
            "/inject",
            {},
            json.dumps({"cloud": "azure", "metric": "*", "count": 2}).encode(),
        )
        assert status == 200
        assert payload["accepted"] > 0
        assert "generated" in payload["note"]

    def test_inject_rejects_unknown_cloud(self, api_pipeline: Pipeline) -> None:
        status, payload = ControlServer(api_pipeline, port=0).handle(
            "POST",
            "/inject",
            {},
            json.dumps({"cloud": "heroku", "metric": "*", "count": 1}).encode(),
        )
        assert status == 400
        assert "unknown cloud" in payload["error"]

    def test_inject_requires_a_metric_name(self, api_pipeline: Pipeline) -> None:
        status, _ = ControlServer(api_pipeline, port=0).handle("POST", "/inject", {}, json.dumps({}).encode())
        assert status == 400

    def test_url_uses_loopback_for_wildcard_bind(self, api_pipeline: Pipeline) -> None:
        assert ControlServer(api_pipeline, host="0.0.0.0", port=9).url == "http://127.0.0.1:9"


class TestAuthentication:
    """The control API can inject incidents and resolve them - it needs a token."""

    @pytest.fixture
    def secured(self, api_pipeline: Pipeline):
        return ControlServer(api_pipeline, host="127.0.0.1", port=0, token="s3cret-token")

    def test_open_when_no_token_is_configured(self, api_pipeline: Pipeline) -> None:
        server = ControlServer(api_pipeline, port=0)
        assert server.auth_required is False
        assert server.authenticate(None)[0] is True

    def test_health_is_always_public(self, secured) -> None:
        status, payload = secured.handle("GET", "/health", {}, b"", None)
        assert status == 200
        assert payload["auth"] == "required"
        assert secured.rejected_requests == 0

    @pytest.mark.parametrize("path", ["/stats", "/metrics", "/events", "/incidents"])
    def test_protected_routes_reject_missing_credentials(self, secured, path: str) -> None:
        status, payload = secured.handle("GET", path, {}, b"", None)
        assert status == 401
        assert "Authorization" in payload["error"]
        assert secured.rejected_requests == 1

    def test_wrong_token_is_rejected(self, secured) -> None:
        assert secured.handle("GET", "/stats", {}, b"", "Bearer wrong")[0] == 401
        assert secured.handle("GET", "/stats", {}, b"", "Bearer s3cret-toke")[0] == 401
        assert secured.handle("GET", "/stats", {}, b"", "Basic s3cret-token")[0] == 401
        assert secured.handle("GET", "/stats", {}, b"", "s3cret-token")[0] == 401

    def test_surrounding_whitespace_is_tolerated(self, secured) -> None:
        """RFC 6750 treats whitespace around the credential as transport noise."""
        assert secured.handle("GET", "/stats", {}, b"", "Bearer   s3cret-token  ")[0] == 200

    def test_correct_token_is_accepted(self, secured) -> None:
        status, _ = secured.handle("GET", "/stats", {}, b"", "Bearer s3cret-token")
        assert status == 200
        assert secured.rejected_requests == 0

    def test_scheme_is_case_insensitive(self, secured) -> None:
        assert secured.handle("GET", "/stats", {}, b"", "bearer s3cret-token")[0] == 200

    def test_injection_is_protected_too(self, secured) -> None:
        body = json.dumps({"cloud": "aws", "metric": "cpu_utilization", "value": 99.0}).encode()
        assert secured.handle("POST", "/inject", {}, body, None)[0] == 401
        assert secured.handle("POST", "/inject", {}, body, "Bearer s3cret-token")[0] == 200

    def test_over_http(self, secured) -> None:
        secured.start()
        try:
            status, _ = _call(secured, "GET", "/stats")
            assert status == 401
            status, payload = _call(secured, "GET", "/stats", token="s3cret-token")
            assert status == 200
            assert "summary" in payload
            status, _ = _call(secured, "GET", "/health", token="s3cret-token")
            assert status == 200
        finally:
            secured.stop()


class TestHttpRoundTrip:
    def test_health_over_http(self, server: ControlServer) -> None:
        status, payload = _call(server, "GET", "/health")
        assert status == 200
        assert payload["status"] == "ok"

    def test_full_incident_lifecycle_over_http(self, server: ControlServer) -> None:
        status, injected = _call(
            server,
            "POST",
            "/inject",
            {"cloud": "aws", "metric": "cpu_utilization", "value": 99.0},
        )
        assert status == 200
        incident = injected["incidents"][0]
        assert incident["severity"] == "CRITICAL"

        status, listed = _call(server, "GET", "/incidents?status=open")
        assert incident["id"] in {item["id"] for item in listed["incidents"]}

        status, acked = _call(server, "POST", f"/incidents/{incident['id']}/ack")
        assert acked["incident"]["status"] == "acknowledged"

        status, resolved = _call(server, "POST", f"/incidents/{incident['id']}/resolve")
        assert resolved["incident"]["status"] == "resolved"

        status, feed = _call(server, "GET", "/events?limit=5")
        assert {event["kind"] for event in feed["events"]} >= {"created", "acknowledged", "resolved"}

    def test_stats_over_http(self, server: ControlServer) -> None:
        status, payload = _call(server, "GET", "/stats")
        assert status == 200
        assert payload["summary"]["detector"]["fitted"] is True

    def test_metrics_endpoint(self, server: ControlServer) -> None:
        api_pipeline = server.pipeline
        api_pipeline.tick()
        status, payload = _call(server, "GET", "/metrics")
        assert status == 200
        assert payload["metrics_seen"] == 55

    def test_404_over_http(self, server: ControlServer) -> None:
        status, payload = _call(server, "GET", "/does-not-exist")
        assert status == 404
        assert "error" in payload

    def test_bad_body_over_http(self, server: ControlServer) -> None:
        request = Request(f"{server.url}/inject", data=b"{", method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as excinfo:
            urlopen(request, timeout=5)
        assert excinfo.value.code == 400

    def test_oversized_body_is_rejected(self, server: ControlServer) -> None:
        huge = json.dumps({"padding": "x" * 1_100_000}).encode()
        request = Request(f"{server.url}/inject", data=huge, method="POST")
        with pytest.raises(HTTPError) as excinfo:
            urlopen(request, timeout=5)
        assert excinfo.value.code == 413

    def test_context_manager_starts_and_stops(self, api_pipeline: Pipeline) -> None:
        with ControlServer(api_pipeline, port=0) as instance:
            assert instance.url.startswith("http://127.0.0.1:")
            status, _ = _call(instance, "GET", "/health")
            assert status == 200
        with pytest.raises(URLError):
            urlopen(f"{instance.url}/health", timeout=2)
