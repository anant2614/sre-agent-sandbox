"""Integration tests for the FastAPI server.

Tests are organized by endpoint / feature area:
  - GET /health
  - GET /schema
  - POST /reset
  - POST /step
  - GET /state
  - POST /step before POST /reset (409)
  - WebSocket /ws lifecycle
  - WebSocket error handling
  - Concurrent WebSocket sessions
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sre_agent_sandbox.server.app import app

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

    def test_health_returns_status_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert data == {"status": "ok"}

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

    def test_schema_has_action_and_observation(self, client: TestClient) -> None:
        data = client.get("/schema").json()
        assert "action_space" in data
        assert "observation_space" in data

    def test_schema_action_has_properties(self, client: TestClient) -> None:
        data = client.get("/schema").json()
        action_schema = data["action_space"]
        assert "properties" in action_schema
        assert "action_type" in action_schema["properties"]
        assert "target_service" in action_schema["properties"]

    def test_schema_observation_has_properties(self, client: TestClient) -> None:
        data = client.get("/schema").json()
        obs_schema = data["observation_space"]
        assert "properties" in obs_schema
        assert "metrics" in obs_schema["properties"]
        assert "health_status" in obs_schema["properties"]


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------


class TestResetEndpoint:
    """VAL-SRV-003: Reset endpoint returns valid observation."""

    def test_reset_returns_200(self, client: TestClient) -> None:
        resp = client.post("/reset")
        assert resp.status_code == 200

    def test_reset_returns_observation_fields(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        assert "metrics" in data
        assert "log_buffer" in data
        assert "health_status" in data
        assert "active_alerts" in data
        assert "done" in data
        assert "reward" in data

    def test_reset_done_is_false(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        assert data["done"] is False

    def test_reset_all_services_healthy(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        for svc in ("api", "order", "db"):
            assert data["health_status"][svc] is True

    def test_reset_metrics_has_all_services(self, client: TestClient) -> None:
        data = client.post("/reset").json()
        for svc in ("api", "order", "db"):
            assert svc in data["metrics"]
            for key in ("cpu", "memory", "latency", "request_count"):
                assert key in data["metrics"][svc]

    def test_reset_with_seed(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is False

    def test_reset_with_episode_id(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"episode_id": "my-episode"})
        assert resp.status_code == 200

    def test_reset_with_seed_and_episode_id(self, client: TestClient) -> None:
        resp = client.post("/reset", json={"seed": 42, "episode_id": "ep-42"})
        assert resp.status_code == 200

    def test_reset_empty_body_ok(self, client: TestClient) -> None:
        """POST /reset with no body should work."""
        resp = client.post("/reset")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------


class TestStepEndpoint:
    """VAL-SRV-004: Step endpoint accepts action and returns observation+reward."""

    def test_step_returns_200_after_reset(self, client: TestClient) -> None:
        client.post("/reset")
        resp = client.post("/step", json={"action": {"action_type": 0, "target_service": "api"}})
        assert resp.status_code == 200

    def test_step_returns_all_fields(self, client: TestClient) -> None:
        client.post("/reset")
        data = client.post(
            "/step", json={"action": {"action_type": 0, "target_service": "api"}}
        ).json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "terminated" in data
        assert "truncated" in data
        assert "info" in data

    def test_step_done_field_matches_terminated_or_truncated(self, client: TestClient) -> None:
        """done = terminated or truncated."""
        client.post("/reset")
        data = client.post(
            "/step", json={"action": {"action_type": 0, "target_service": "api"}}
        ).json()
        assert data["done"] == (data["terminated"] or data["truncated"])

    def test_step_done_is_bool(self, client: TestClient) -> None:
        client.post("/reset")
        data = client.post(
            "/step", json={"action": {"action_type": 0, "target_service": "api"}}
        ).json()
        assert isinstance(data["done"], bool)

    def test_step_observation_has_correct_structure(self, client: TestClient) -> None:
        client.post("/reset")
        data = client.post(
            "/step", json={"action": {"action_type": 1, "target_service": "db"}}
        ).json()
        obs = data["observation"]
        assert "metrics" in obs
        assert "health_status" in obs
        assert "log_buffer" in obs
        assert "active_alerts" in obs

    def test_step_reward_is_numeric(self, client: TestClient) -> None:
        client.post("/reset")
        data = client.post(
            "/step", json={"action": {"action_type": 0, "target_service": "api"}}
        ).json()
        assert isinstance(data["reward"], (int, float))

    def test_step_terminated_is_bool(self, client: TestClient) -> None:
        client.post("/reset")
        data = client.post(
            "/step", json={"action": {"action_type": 0, "target_service": "api"}}
        ).json()
        assert isinstance(data["terminated"], bool)

    def test_step_multiple_steps(self, client: TestClient) -> None:
        """Multiple steps should all succeed."""
        client.post("/reset")
        for _ in range(5):
            resp = client.post(
                "/step", json={"action": {"action_type": 0, "target_service": "api"}}
            )
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /step before POST /reset → 409
# ---------------------------------------------------------------------------


class TestStepBeforeReset:
    """VAL-SRV-005: Step before reset returns error."""

    def test_step_without_reset_returns_409(self, client: TestClient) -> None:
        resp = client.post(
            "/step", json={"action": {"action_type": 0, "target_service": "api"}}
        )
        assert resp.status_code == 409

    def test_step_without_reset_has_error_detail(self, client: TestClient) -> None:
        resp = client.post(
            "/step", json={"action": {"action_type": 0, "target_service": "api"}}
        )
        data = resp.json()
        assert "detail" in data


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------


class TestStateEndpoint:
    """VAL-SRV-008: GET /state returns current state."""

    def test_state_after_reset(self, client: TestClient) -> None:
        client.post("/reset")
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_id" in data
        assert "step_count" in data
        assert "system_health_score" in data
        assert "active_incidents" in data

    def test_state_step_count_matches(self, client: TestClient) -> None:
        client.post("/reset")
        client.post("/step", json={"action": {"action_type": 0, "target_service": "api"}})
        client.post("/step", json={"action": {"action_type": 0, "target_service": "api"}})
        data = client.get("/state").json()
        assert data["step_count"] == 2

    def test_state_before_reset_returns_409(self, client: TestClient) -> None:
        """GET /state without prior reset should return error."""
        resp = client.get("/state")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Session independence (REST)
# ---------------------------------------------------------------------------


class TestRestSessionIndependence:
    """Each REST TestClient gets its own SREEnvironment."""

    def test_separate_clients_independent(self) -> None:
        client1 = TestClient(app)
        client2 = TestClient(app)

        # Reset both
        client1.post("/reset", json={"seed": 1})
        client2.post("/reset", json={"seed": 2})

        # Step client1, leave client2 alone
        client1.post("/step", json={"action": {"action_type": 0, "target_service": "api"}})

        # client1 should show step_count=1, client2 step_count=0
        state1 = client1.get("/state").json()
        state2 = client2.get("/state").json()
        assert state1["step_count"] == 1
        assert state2["step_count"] == 0


# ---------------------------------------------------------------------------
# WebSocket /ws — lifecycle
# ---------------------------------------------------------------------------


class TestWebSocketLifecycle:
    """VAL-SRV-006: WebSocket session lifecycle."""

    def test_ws_reset(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            resp = ws.receive_json()
            assert resp["type"] == "observation"
            assert "data" in resp
            assert resp["data"]["done"] is False

    def test_ws_step_after_reset(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()  # observation from reset

            ws.send_json({
                "type": "step",
                "data": {"action": {"action_type": 0, "target_service": "api"}},
            })
            resp = ws.receive_json()
            assert resp["type"] == "step_result"
            assert "observation" in resp["data"]
            assert "reward" in resp["data"]
            assert "done" in resp["data"]
            assert "terminated" in resp["data"]

    def test_ws_step_done_matches_terminated_or_truncated(self, client: TestClient) -> None:
        """WebSocket step_result done = terminated or truncated."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset"})
            ws.receive_json()

            ws.send_json({
                "type": "step",
                "data": {"action": {"action_type": 0, "target_service": "api"}},
            })
            resp = ws.receive_json()
            data = resp["data"]
            assert data["done"] == (data["terminated"] or data["truncated"])

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
            # Connection should close cleanly — receiving should fail
            # or return a close message

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
                "data": {"action": {"action_type": 1, "target_service": "db"}},
            })
            step_resp = ws.receive_json()
            assert step_resp["type"] == "step_result"

            # State
            ws.send_json({"type": "state"})
            state_resp = ws.receive_json()
            assert state_resp["type"] == "state"
            assert state_resp["data"]["step_count"] == 1


# ---------------------------------------------------------------------------
# WebSocket error handling
# ---------------------------------------------------------------------------


class TestWebSocketErrorHandling:
    """VAL-SRV-007: WebSocket error handling."""

    def test_ws_malformed_json(self, client: TestClient) -> None:
        """Malformed JSON returns error, keeps connection alive."""
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not valid json{{{")
            resp = ws.receive_json()
            assert resp["type"] == "error"

            # Connection still alive — can still send valid commands
            ws.send_json({"type": "reset"})
            obs = ws.receive_json()
            assert obs["type"] == "observation"

    def test_ws_unknown_type(self, client: TestClient) -> None:
        """Unknown message type returns error, keeps connection alive."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "foobar"})
            resp = ws.receive_json()
            assert resp["type"] == "error"

            # Connection still alive
            ws.send_json({"type": "reset"})
            obs = ws.receive_json()
            assert obs["type"] == "observation"

    def test_ws_step_before_reset(self, client: TestClient) -> None:
        """Step before reset over WebSocket returns error, keeps connection."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "step",
                "data": {"action": {"action_type": 0, "target_service": "api"}},
            })
            resp = ws.receive_json()
            assert resp["type"] == "error"

            # Can still reset and continue
            ws.send_json({"type": "reset"})
            obs = ws.receive_json()
            assert obs["type"] == "observation"

    def test_ws_state_before_reset(self, client: TestClient) -> None:
        """State before reset over WebSocket returns error."""
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "state"})
            resp = ws.receive_json()
            assert resp["type"] == "error"


# ---------------------------------------------------------------------------
# Concurrent WebSocket sessions
# ---------------------------------------------------------------------------


class TestConcurrentWebSocket:
    """VAL-SRV-009: Concurrent WebSocket sessions are independent."""

    def test_two_sessions_independent(self, client: TestClient) -> None:
        """Two WS connections maintain independent state."""
        with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
            # Reset both with different seeds
            ws1.send_json({"type": "reset", "data": {"seed": 1}})
            ws2.send_json({"type": "reset", "data": {"seed": 2}})
            ws1.receive_json()
            ws2.receive_json()

            # Step ws1 only
            ws1.send_json({
                "type": "step",
                "data": {"action": {"action_type": 0, "target_service": "api"}},
            })
            ws1.receive_json()

            # Check states
            ws1.send_json({"type": "state"})
            ws2.send_json({"type": "state"})
            state1 = ws1.receive_json()
            state2 = ws2.receive_json()

            assert state1["data"]["step_count"] == 1
            assert state2["data"]["step_count"] == 0

    def test_actions_in_one_session_dont_affect_other(self, client: TestClient) -> None:
        """Actions in session 1 don't change observations in session 2."""
        with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
            # Reset both with same seed
            ws1.send_json({"type": "reset", "data": {"seed": 42}})
            ws2.send_json({"type": "reset", "data": {"seed": 42}})
            ws1.receive_json()
            ws2.receive_json()

            # Take different actions in ws1
            for _ in range(5):
                ws1.send_json({
                    "type": "step",
                    "data": {"action": {"action_type": 1, "target_service": "api"}},
                })
                ws1.receive_json()

            # ws2 state should be untouched
            ws2.send_json({"type": "state"})
            state2 = ws2.receive_json()
            assert state2["data"]["step_count"] == 0
