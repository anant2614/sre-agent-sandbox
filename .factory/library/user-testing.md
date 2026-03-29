# User Testing

Testing surface, resource cost classification, and validation approach.

---

## Validation Surface
- **Terminal/CLI**: Run demo script, observe ASCII dashboard output
- **API endpoints**: curl to REST endpoints (reset, step, state, health, schema)
- **Docker**: Build image, run container, verify connectivity
- No browser needed — all validation is API/terminal based

## Validation Tools
- pytest (unit + integration tests)
- curl (REST endpoint testing)
- Python websockets library (WebSocket testing)
- docker CLI (build/run/stop)

## Validation Concurrency
- **Max concurrent validators: 5**
- Rationale: Each validator is a lightweight Python process (~100MB RAM). Machine has 16GB RAM, 10 cores, ~6GB baseline usage. Headroom: 10GB * 0.7 = 7GB. 5 validators = ~500MB — well within budget.
- Docker tests should be serialized (single container on port 8000)

## Known Constraints
- Docker must be running for Docker-related validation
- Port 8000 must be free for server/Docker tests
- Environment step() is purely in-memory, no external services
