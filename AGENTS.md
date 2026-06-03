# AGENTS.md

## Commands

- **Backend setup:** `python3 -m venv venv-classidyne && source venv-classidyne/bin/activate && pip install -r requirements.txt`
- **Backend run:** `python app.py` — starts uvicorn, auto-picks a port between 5000–5005
- **Backend test (using existing `venv`):** `source venv/bin/activate && python -m pytest tests/ -v` — tests connect to the running server (default port 5000; override with `CLASSIDYNE_PORT`)
- **Frontend setup:** `cd frontend && npm ci`
- **Frontend dev:** `cd frontend && npm start`
- **Frontend build:** `cd frontend && npm run build` — output goes to `frontend/build`, NOT `static/` (must sync manually)
- **Frontend test:** `cd frontend && CI=true npm test -- --runInBand --watch=false`
- **Frontend single test:** `cd frontend && CI=true npm test -- --runInBand src/App.test.tsx --watch=false`
- **No standalone lint** — CRA runs ESLint during `npm start` and `npm run build`

## Prerequisites

- **Git LFS is mandatory.** `RadioNet/RadioNet.pth` is LFS-tracked. Without `git lfs pull`, backend import will fail at module level before any request is served.
- **Kaggle dataset** must be downloaded and unzipped into `datasets/` with structure `datasets/{waterfall,fft}/<signal-type>/` before classification works. A fresh checkout cannot classify anything until embedding is run.

## Architecture

- **Backend:** Single FastAPI app in `app.py`. Loads `RadioNetExtractor` (ResNet-34 via `timm`) and initializes ChromaDB (`classidyne_db/`) as module-level singletons at import time.
- **Two vector collections:** `waterfall` and `fft`. Most endpoints accept a `collection` parameter restricted to those two values.
- **Embedding pipeline:** Walks `datasets/waterfall/` and `datasets/fft/`, hashes each image (duplicate check is hash-based, not filename-based), extracts embeddings, upserts into ChromaDB. Runs as a background task via `/api/start-embedding`.
- **Frequency data:** `known_frequencies.json` is the sole source of frequency metadata — not stored in or derived from the vector DB.
- **Frontend:** React + MUI + React Router + React Query in `frontend/src/`. Communicates via relative `/api/...` fetches. No typed API client.
- **Static serving:** `app.py` mounts `static/` as the root. Frontend changes require `npm run build` then copying `frontend/build/` into `static/` — they do NOT auto-update.

## Tests

- **Backend integration tests** live in `tests/`. They connect to the running server (default port 5000) and exercise the API over HTTP.
- **Run with pytest:** `source venv/bin/activate && python -m pytest tests/ -v` (set `CLASSIDYNE_PORT` to override the default port).
- **Run standalone:** `python tests/test_api.py --host localhost --port 5000` (no pytest required).
- **Fixtures:** `tests/conftest.py` provides an `httpx.Client` fixture (`client`) wired to `http://localhost:<port>` with a 30 s timeout.
- **Test image:** `tests/lora.png` is used for classification tests.
- **What's covered:** stats, waterfall/fft type listing, classification (waterfall & fft), image lookup, frequency identification, and collage generation.

## Key Conventions

- All API responses use `{"success": bool, "message": str, ...}` shape, even on errors with HTTP status codes. Preserve this.
- Image lookup/delete uses a strict priority: exact Chroma ID/hash → exact filepath → partial basename match.
- Classification preprocessing converts to grayscale then back to RGB (`convert("L").convert("RGB")`) — keep this aligned if modifying.
- The virtualenv is named `venv-classidyne` (not `venv`). You must activate it before running the backend.
- To reset the database, delete `classidyne_db/` before re-embedding. Deleting from the API does NOT remove the source file from `datasets/`.
- **Test fixtures:** `tests/conftest.py` provides a `client` fixture that connects to `http://localhost:5000` (the running backend) for pytest. Update this port if the backend runs elsewhere.
