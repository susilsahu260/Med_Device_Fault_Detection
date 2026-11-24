# Plan: FastAPI Backend — V1

## Implementation Order (dependency-aware)

```
schemas.py  ──→  api.py  ──→  test_api.py
   │                │
   ▼                ▼
Physical-bound     SQLite
Pydantic v2        persistence
validators         (predictions.db)
```

## Tasks

- [x] SPEC written: `spec/SPEC-fastapi-backend.md`
- [ ] T1: Install `fastapi[standard]`, `uvicorn[standard]`, `httpx` into `.venv`
- [ ] T2: Write `src/schemas.py` — Pydantic v2 request/response models
  - Acceptance: All 16 features validated, NaN/Inf rejected, bounds enforced, 422 on wrong window length
  - Verify: `pytest tests/test_api.py::test_predict_wrong_window_length -v`
  - Files: `src/schemas.py`
- [ ] T3: Write `src/api.py` — FastAPI app with lifespan, 3 endpoints, SQLite
  - Acceptance: Lifespan loads model/artifacts into app.state; /health returns model_loaded; /predict runs real inference; /history queries SQLite
  - Verify: `pytest tests/test_api.py -v`
  - Files: `src/api.py`
- [ ] T4: Write `tests/test_api.py` — 11 pytest tests
  - Acceptance: All tests pass; TestClient uses lifespan context manager
  - Verify: `pytest tests/test_api.py -v --tb=short`
  - Files: `tests/test_api.py`
- [ ] T5: Run full suite, fix any failures
  - Verify: `pytest -v`
- [ ] T6: Smoke-test live server (start uvicorn, curl 3 endpoints, stop)
  - Verify: All 3 endpoints return expected JSON

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| TF load time slows lifespan startup in tests | Load model once per TestClient session using `with TestClient(app)` |
| SQLite file persists between test runs contaminating history | Use `tmp_path` fixture or dedicated test DB path via env var override |
| Pydantic v2 `field_validator` mode differences | Use `mode="before"` for NaN/Inf; `mode="after"` for cross-field checks |
| `preprocessing.py` requires DataFrame input but API receives JSON | Build a DataFrame from `readings` list inside `/predict` route |
| `best_lstm_model.keras` missing → 500 at startup | Health endpoint returns `model_loaded: false` gracefully if file missing |
