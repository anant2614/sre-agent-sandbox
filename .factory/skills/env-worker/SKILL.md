---
name: env-worker
description: Implements core environment logic — models, chaos engine, reward, simulated system, and the SREEnvironment class
---

# Environment Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Features involving:
- Pydantic model definitions (SREAction, SREObservation, SREState)
- Simulated 3-tier system state model
- ChaosEngine (fault injection logic)
- Reward computation
- SREEnvironment class (reset/step/state)
- openenv.yaml manifest
- Unit tests for all of the above

## Work Procedure

1. **Read context**: Read `AGENTS.md`, `.factory/research/openenv-api.md`, and `.factory/library/architecture.md` to understand the OpenEnv API and project structure.

2. **Write tests first (TDD)**:
   - Create or update the relevant test file in `tests/`
   - Write failing tests that cover the feature's expectedBehavior and verificationSteps
   - Tests should import from the target module and assert expected behavior
   - Run `uv run pytest tests/ -v --tb=short -x` — tests should FAIL (red phase)

3. **Implement the feature**:
   - Create or update the source module
   - Follow OpenEnv conventions: subclass Action/Observation/State from `openenv.core.env_server.types`
   - Use seeded `random.Random(seed)` for all randomness — never module-level random
   - All simulation is in-memory, no I/O
   - Pydantic v2 with `extra="forbid"` on Action/Observation, `extra="allow"` on State

4. **Run tests (green phase)**:
   - `uv run pytest tests/ -v --tb=short -x` — all tests must pass
   - Fix any failures before proceeding

5. **Run full test suite**:
   - `uv run pytest tests/ -v --tb=short` — verify no regressions across all test files

6. **Manual verification**:
   - Import the module in a Python shell and exercise the key behaviors
   - For environment features: call reset(), step() with various actions, verify observation structure
   - For chaos features: inject faults, observe metric changes
   - For reward: compute rewards for known scenarios, verify against expected values

7. **Update library** if you discover important patterns or gotchas: `.factory/library/architecture.md`

## Example Handoff

```json
{
  "salientSummary": "Implemented SREAction, SREObservation, SREState Pydantic models with OpenEnv base classes. Added Pydantic validators for action_type (0-4) and target_service ('api','order','db'). Ran `uv run pytest tests/test_models.py -v` (12 passing) and verified serialization round-trip in Python shell.",
  "whatWasImplemented": "Three Pydantic v2 models: SREAction(Action) with action_type IntEnum and target_service Literal validator, SREObservation(Observation) with metrics dict, log_buffer list, health_status dict, active_alerts list, SREState(State) with system_health_score float and active_incidents list. All models use extra='forbid' except State which uses extra='allow' per OpenEnv convention.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv run pytest tests/test_models.py -v --tb=short", "exitCode": 0, "observation": "12 tests passed: valid construction, invalid action_type rejected, invalid target rejected, extra fields rejected, serialization round-trip, all 15 valid combos accepted"},
      {"command": "uv run python -c \"from sre_agent_sandbox.models import SREAction; a = SREAction(action_type=1, target_service='api'); print(a.model_dump_json())\"", "exitCode": 0, "observation": "Printed valid JSON with action_type and target_service fields"}
    ],
    "interactiveChecks": [
      {"action": "Constructed SREAction with all valid combos (0-4 x 3 services)", "observed": "All 15 combinations created without error"},
      {"action": "Attempted SREAction(action_type=5, target_service='api')", "observed": "ValidationError raised as expected"},
      {"action": "Attempted SREAction(action_type=0, target_service='api', extra='bad')", "observed": "ValidationError raised due to extra='forbid'"}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/test_models.py", "cases": [
        {"name": "test_valid_action_construction", "verifies": "All 15 action-target combos accepted"},
        {"name": "test_invalid_action_type_rejected", "verifies": "Out of range action_type raises ValidationError"},
        {"name": "test_extra_fields_rejected", "verifies": "extra='forbid' enforced on Action and Observation"},
        {"name": "test_serialization_roundtrip", "verifies": "JSON serialize->deserialize produces equal model"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- OpenEnv base class imports fail (package not installed correctly)
- Feature depends on another module that doesn't exist yet
- Requirements are ambiguous (e.g., unclear fault behavior)
- Test infrastructure issues (pytest not working)
