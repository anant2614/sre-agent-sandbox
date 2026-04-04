"""Integration tests for the OpenEnv-standard FastAPI server.

The server is built via ``create_app()`` from the openenv library.  Key
behavioural difference from a hand-rolled server:

* **REST endpoints are stateless.**  ``POST /reset`` and ``POST /step`` each
  spin up a *fresh* environment instance.  There is no shared state across
  REST calls.  Stateful multi-step interaction requires the WebSocket
  endpoint (``/ws``).
* **WebSocket sessions are stateful.**  Each ``/ws`` connection gets its
  own dedicated ``SREEnvironment`` instance that persists for the lifetime
  of the connection.

Tests are organized by:
  - GET /health
  - GET /schema
  - POST /reset  (stateless)
  - POST /step   (stateless — one-shot)
  - WebSocket /ws lifecycle
  - WebSocket error handling
  - Concurrent WebSocket sessions
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Fresh TestClient for each test — independent session."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """VAL-SRV-001: Health endpoint returns 200."""

    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status_healthy(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"

    def test_health_content_type_json(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# GET /schema
# ---------------------------------------------------------------------------


class TestSchemaEndpoint:
    """VAL-SRV-002: Schema endpoint returns valid JSON schemas."""

    def test_schema_returns_200(self, client: TestClient) -> None:
        resp = client.get("/schema")
        assert resp.status_code == 200

    def test_schema_has_action_observation_state(self, client: TestClient) -> None:
        data = client.get("/schema").json()
        assert "action" in data
        assert "observation" in data
        assert "state" in data

    def test_schema_action_has_properties(self, client: TestClient) -> None:
        data = client.get("/schema").json()
        action_schema = data["action"]
        assert "properties" in action_schema
        assert "action_type" in action_schema["properties"]
        assert "target_service" in action_schema["properties"]

    def test_schema_observation_has_properties(self, client: TestClient) -> None:
        data = client.get("/schema").json()
        obs_schema = data["observation"]
        assert "properties" in obs_schema
        assert "metrics" in obs_schema["properties"]
        assert "health_status" in obs_schema["properties"]


# ---------------------------------------------------------------------------
# POST /reset  (stateless — creates fresh env, resets, returns obs, closes)
# ---------------------------------------------------------------------------


class TestResetEndpoint:
    """VAL-SRV-003: Reset endpoint returns valid observation."""

    def test_reset_returns_200(self, client: TestClient) -> None:
        resp = client.post("/reset")
        assert resp.status_code == 200

    def test_reset_returns_observation_fields(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        assert "observation" in data
        obs = data["observation"]
        assert "metrics" in obs
        assert "log_buffer" in obs
        assert "health_status" in obs
        assert "active_alerts" in obs
        assert "done" in data
        assert "reward" in data

    def test_reset_done_is_false(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        assert data["done"] is False

    def test_reset_all_services_healthy(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        obs = data["observation"]
        for svc in ("api", "order", "db"):
            assert obs["health_status"][svc] is True

    def test_reset_metrics_has_all_services(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        obs = data["observation"]
        for svc in ("api", "order", "db"):
            assert svc in obs["metrics"]
            for key in ("cpu", "memory", "latency", "request_count"):
                assert key in obs["metrics"][svc]

    def test_reset_with_seed(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is False

    def test_reset_empty_body_ok(self, client: TestClient) -> None:
        resp = client.post("/reset")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /step (stateless — one-shot, env NOT pre-reset by create_app)
# ---------------------------------------------------------------------------


class TestStepEndpoint:
    """VAL-SRV-004: Step endpoint (stateless REST mode).

    With ``create_app``, REST ``POST /step`` creates a fresh environment.
    Multi-step interaction must use WebSocket.  These tests only verify
    schema-level validation.
    """

    def test_step_rejects_invalid_action_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/step", json={"action": {"action_type": 99, "target_service": "api"}}
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# WebSocket /ws — lifecycle  (stateful)
# ---------------------------------------------------------------------------


class TestWebSocketLifecycle:
    """VAL-SRV-006: WebSocket session lifecycle."""

    def test_ws_reset(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            resp = ws.receive_json()
            assert resp["type"] == "observation"
            assert "data" in resp
            obs = resp["data"]
            assert obs["done"] is False
            assert "observation" in obs

    def test_ws_step_after_reset(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json({
                "type": "step",
                "data": {"action_type": 0, "target_service": "api"},
            })
            resp = ws.receive_json()
            assert resp["type"] == "observation"
            data = resp["data"]
            assert "observation" in data
            assert "reward" in data
            assert "done" in data

    def test_ws_step_done_is_bool(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json({
                "type": "step",
                "data": {"action_type": 0, "target_service": "api"},
            })
            resp = ws.receive_json()
            assert isinstance(resp["data"]["done"], bool)

    def test_ws_state(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json({"type": "state"})
            resp = ws.receive_json()
            assert resp["type"] == "state"
            assert "episode_id" in resp["data"]
            assert "step_count" in resp["data"]

    def test_ws_close(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()
            ws.send_json({"type": "close"})

    def test_ws_full_lifecycle(self, client: TestClient) -> None:
        """Full lifecycle: reset -> step -> state -> close."""
        with client.websocket_connect("/ws") as ws:
            # Reset
            ws.send_json({"type": "reset"})
            obs_resp = ws.receive_json()
            assert obs_resp["type"] == "observation"

            # Step
            ws.send_json({
                "type": "step",
                "data": {"action_type": 1, "target_service": "db"},
            })
            step_resp = ws.receive_json()
            assert step_resp["type"] == "observation"

            # State
            ws.send_json({"type": "state"})
            state_resp = ws.receive_json()
            assert state_resp["type"] == "state"
            assert state_resp["data"]["step_count"] == 1

    def test_ws_step_observation_structure(self, client: TestClient) -> None:
        """Verify observation has the expected SRE fields."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json({
                "type": "step",
                "data": {"action_type": 1, "target_service": "db"},
            })
            resp = ws.receive_json()
            obs = resp["data"]["observation"]
            assert "metrics" in obs
            assert "health_status" in obs
            assert "log_buffer" in obs
            assert "active_alerts" in obs

    def test_ws_step_reward_is_numeric(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json({
                "type": "step",
                "data": {"action_type": 0, "target_service": "api"},
            })
            resp = ws.receive_json()
            assert isinstance(resp["data"]["reward"], (int, float))

    def test_ws_multiple_steps(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            for _ in range(5):
                ws.send_json({
                    "type": "step",
                    "data": {"action_type": 0, "target_service": "api"},
                })
                resp = ws.receive_json()
                assert resp["type"] == "observation"

    def test_ws_state_step_count_matches(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            for _ in range(3):
                ws.send_json({
                    "type": "step",
                    "data": {"action_type": 0, "target_service": "api"},
                })
                ws.receive_json()

            ws.send_json({"type": "state"})
            state = ws.receive_json()
            assert state["data"]["step_count"] == 3


# ---------------------------------------------------------------------------
# WebSocket error handling
# ---------------------------------------------------------------------------


class TestWebSocketErrorHandling:
    """VAL-SRV-007: WebSocket error handling."""

    def test_ws_malformed_json(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not valid json{{{")
            resp = ws.receive_json()
            assert resp["type"] == "error"

            ws.send_json({"type": "reset"})
            obs = ws.receive_json()
            assert obs["type"] == "observation"

    def test_ws_unknown_type(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "foobar"})
            resp = ws.receive_json()
            assert resp["type"] == "error"

            ws.send_json({"type": "reset"})
            obs = ws.receive_json()
            assert obs["type"] == "observation"

    def test_ws_step_before_reset(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "step",
                "data": {"action_type": 0, "target_service": "api"},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"

            ws.send_json({"type": "reset"})
            obs = ws.receive_json()
            assert obs["type"] == "observation"

    def test_ws_state_before_reset(self, client: TestClient) -> None:
        """State before reset should return error or default state."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "state"})
            resp = ws.receive_json()
            # create_app returns state even before reset (env exists, state property works)
            assert resp["type"] in ("state", "error")


# ---------------------------------------------------------------------------
# Concurrent WebSocket sessions
# ---------------------------------------------------------------------------


class TestConcurrentWebSocket:
    """VAL-SRV-009: Concurrent WebSocket sessions are independent."""

    def test_two_sessions_independent(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
            ws1.send_json({"type": "reset", "data": {"seed": 1}})
            ws2.send_json({"type": "reset", "data": {"seed": 2}})
            ws1.receive_json()
            ws2.receive_json()

            ws1.send_json({
                "type": "step",
                "data": {"action_type": 0, "target_service": "api"},
            })
            ws1.receive_json()

            ws1.send_json({"type": "state"})
            ws2.send_json({"type": "state"})
            state1 = ws1.receive_json()
            state2 = ws2.receive_json()

            assert state1["data"]["step_count"] == 1
            assert state2["data"]["step_count"] == 0

    def test_actions_in_one_session_dont_affect_other(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
            ws1.send_json({"type": "reset", "data": {"seed": 42}})
            ws2.send_json({"type": "reset", "data": {"seed": 42}})
            ws1.receive_json()
            ws2.receive_json()

            for _ in range(5):
                ws1.send_json({
                    "type": "step",
                    "data": {"action_type": 1, "target_service": "api"},
                })
                ws1.receive_json()

            ws2.send_json({"type": "state"})
            state2 = ws2.receive_json()
            assert state2["data"]["step_count"] == 0
