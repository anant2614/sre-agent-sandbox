"""FastAPI server for the SRE Agent Sandbox environment.

Provides REST endpoints (GET /health, GET /schema, POST /reset, POST /step,
GET /state) and a WebSocket endpoint (/ws) for interacting with the
:class:`SREEnvironment`.

Session management: each REST client session and each WebSocket connection
gets its own independent :class:`SREEnvironment` instance.  REST sessions
are tracked via a ``sre_session_id`` cookie.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

from fastapi import Cookie, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from sre_agent_sandbox.models import SREAction, SREObservation
from sre_agent_sandbox.server.environment import SREEnvironment

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SRE Agent Sandbox",
    description="OpenEnv RL environment for SRE agent training",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Optional parameters for POST /reset."""

    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    """Request body for POST /step."""

    action: Dict[str, Any]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

# Map of session_id -> SREEnvironment for REST clients.
_rest_sessions: Dict[str, SREEnvironment] = {}

_SESSION_COOKIE = "sre_session_id"


def _get_or_create_env(session_id: Optional[str]) -> tuple[str, SREEnvironment]:
    """Return (session_id, env) for the given cookie value.

    Creates a new environment (and a new session id) when the cookie is
    absent or unrecognised.
    """
    if session_id and session_id in _rest_sessions:
        return session_id, _rest_sessions[session_id]
    new_id = session_id or str(uuid.uuid4())
    env = SREEnvironment()
    _rest_sessions[new_id] = env
    return new_id, env


def _json_response(data: Any, session_id: str) -> JSONResponse:
    """Build a JSONResponse with the session cookie attached."""
    resp = JSONResponse(content=data)
    resp.set_cookie(key=_SESSION_COOKIE, value=session_id)
    return resp


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    """GET /health — liveness probe."""
    return {"status": "ok"}


@app.get("/schema")
def schema() -> Dict[str, Any]:
    """GET /schema — JSON schemas for SREAction and SREObservation."""
    return {
        "action_space": SREAction.model_json_schema(),
        "observation_space": SREObservation.model_json_schema(),
    }


@app.post("/reset")
def reset(
    body: Optional[ResetRequest] = None,
    sre_session_id: Optional[str] = Cookie(default=None),
) -> JSONResponse:
    """POST /reset — reset environment and return initial observation."""
    sid, env = _get_or_create_env(sre_session_id)
    seed = body.seed if body else None
    episode_id = body.episode_id if body else None
    obs = env.reset(seed=seed, episode_id=episode_id)
    return _json_response(obs.model_dump(), sid)


@app.post("/step")
def step(
    body: StepRequest,
    sre_session_id: Optional[str] = Cookie(default=None),
) -> JSONResponse:
    """POST /step — take an action and return observation + reward.

    Returns 409 if the environment has not been reset yet.
    """
    sid, env = _get_or_create_env(sre_session_id)

    if not env._is_reset:
        raise HTTPException(
            status_code=409,
            detail="Environment has not been reset. Call POST /reset first.",
        )

    try:
        action = SREAction(**body.action)
    except (ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    obs = env.step(action)
    obs_dict = obs.model_dump()

    result = {
        "observation": {
            "metrics": obs_dict["metrics"],
            "log_buffer": obs_dict["log_buffer"],
            "health_status": obs_dict["health_status"],
            "active_alerts": obs_dict["active_alerts"],
        },
        "reward": obs_dict["reward"] if obs_dict["reward"] is not None else 0.0,
        "terminated": obs_dict["done"],
        "truncated": False,
        "info": {},
    }
    return _json_response(result, sid)


@app.get("/state")
def get_state(
    sre_session_id: Optional[str] = Cookie(default=None),
) -> JSONResponse:
    """GET /state — return current SREState as JSON."""
    sid, env = _get_or_create_env(sre_session_id)

    if not env._is_reset:
        raise HTTPException(
            status_code=409,
            detail="Environment has not been reset. Call POST /reset first.",
        )

    return _json_response(env.state.model_dump(), sid)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket /ws — each connection gets its own SREEnvironment.

    Supported message types: reset, step, state, close.
    Malformed JSON or unknown types return ``{type: 'error'}``
    without closing the connection.
    """
    await websocket.accept()
    env = SREEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()

            # Parse JSON
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Malformed JSON"},
                })
                continue

            msg_type = message.get("type")
            msg_data = message.get("data") or {}

            if msg_type == "reset":
                seed = msg_data.get("seed") if msg_data else None
                episode_id = msg_data.get("episode_id") if msg_data else None
                obs = env.reset(seed=seed, episode_id=episode_id)
                await websocket.send_json({
                    "type": "observation",
                    "data": obs.model_dump(),
                })

            elif msg_type == "step":
                if not env._is_reset:
                    await websocket.send_json({
                        "type": "error",
                        "data": {
                            "message": "Environment has not been reset. Send a reset message first.",
                        },
                    })
                    continue

                try:
                    action_data = msg_data.get("action", {})
                    action = SREAction(**action_data)
                except (ValidationError, TypeError) as exc:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": f"Invalid action: {exc}"},
                    })
                    continue

                obs = env.step(action)
                obs_dict = obs.model_dump()
                await websocket.send_json({
                    "type": "step_result",
                    "data": {
                        "observation": {
                            "metrics": obs_dict["metrics"],
                            "log_buffer": obs_dict["log_buffer"],
                            "health_status": obs_dict["health_status"],
                            "active_alerts": obs_dict["active_alerts"],
                        },
                        "reward": obs_dict["reward"] if obs_dict["reward"] is not None else 0.0,
                        "terminated": obs_dict["done"],
                        "truncated": False,
                        "info": {},
                    },
                })

            elif msg_type == "state":
                if not env._is_reset:
                    await websocket.send_json({
                        "type": "error",
                        "data": {
                            "message": "Environment has not been reset. Send a reset message first.",
                        },
                    })
                    continue

                await websocket.send_json({
                    "type": "state",
                    "data": env.state.model_dump(),
                })

            elif msg_type == "close":
                await websocket.close()
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass
