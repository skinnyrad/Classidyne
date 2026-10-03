# AGENTS.md

Classidyne classifies RF signal images (waterfall and FFT) by embedding them with RadioNet (v3: EfficientNet-B0, SupCon-trained) and doing nearest-neighbor search in ChromaDB. Backend is FastAPI; frontend is React. The waterfall dataset is built and the model trained with the scripts in `dataset_tools/`.

## Commands

- **Backend setup:** `python3 -m venv venv-classidyne && source venv-classidyne/bin/activate && pip install -r requirements.txt`
- **Backend run:** `python app.py` — starts uvicorn on `0.0.0.0`, auto-picks the first free port in 5000–5005 (`reload=False`). On macOS, AirPlay Receiver usually holds 5000, so the app lands on 5001
- **Backend test:** `python -m pytest tests/ -v` inside an activated venv (`pip install pytest`; not in requirements) — tests hit the **running** server (default port 5000; override with `CLASSIDYNE_PORT`). Standalone: `python tests/test_api.py --host localhost --port 5000`
- **App evaluation:** `python dataset_tools/eval/app_eval.py --port 5001` — held-out accuracy through the live vector DB, colormap robustness, HTTP latency (writes `docs/APP_EVALUATION.md`)
- **Train / evaluate RadioNet:** see `dataset_tools/README.md` and `docs/DATASET_GUIDE.md` §13
- **Dataset tools setup:** `pip install -r dataset_tools/requirements.txt` (scipy, pyserial, torchvision, pytest) on top of `requirements.txt`
- **Frontend setup:** `cd frontend && npm ci`
- **Frontend dev:** `cd frontend && npm start`
- **Frontend build:** `cd frontend && npm run build` — output goes to `frontend/build`, NOT `static/` (sync manually: copy `frontend/build/*` into `static/`)
- **Frontend test:** `cd frontend && CI=true npm test -- --runInBand --watch=false`
- **Frontend single test:** `cd frontend && CI=true npm test -- --runInBand src/App.test.tsx --watch=false`
- **No standalone lint** — CRA runs ESLint during `npm start` and `npm run build`

## Prerequisites

- **Git LFS is mandatory.** `RadioNet/RadioNet.pth` is LFS-tracked. Without `git lfs pull`, backend import fails at module level before any request is served.
- **Kaggle dataset** must be downloaded and unzipped into `datasets/` as `datasets/{waterfall,fft}/<signal-type>/<image>`. `datasets/` is gitignored. The waterfall side is dataset v3 (24 classes, see `docs/DATASET_V3.md`); `dataset_tools/manifest.csv` lists every v3 image. A fresh checkout cannot classify anything until embedding has run (`POST /api/start-embedding`, or the UI).
- `classidyne_db/` (Chroma persistence) is created on first import and is not checked in.

## Layout

- `app.py` — entire backend (~690 lines): model, DB, helpers, endpoints, static mount, `__main__` port scan
- `API.md` — endpoint documentation; update when changing endpoints
- `known_frequencies.json` — sole source of frequency metadata
- `RadioNet/RadioNet.pth` — LFS model checkpoint: `model_state_dict`, plus `arch` (timm name) and `preprocess` (`"full"`) for v3 checkpoints; checkpoints without them are treated as ResNet-34 with the timm centre-crop transform
- `frontend/src/` — `App.tsx` (routes), `components/Navbar.tsx`, `pages/{SignalClassification,UploadSignalImages,ManageImages,TypeViewer}.tsx`
- `static/` — checked-in built frontend served by FastAPI
- `tests/` — `conftest.py`, `test_api.py`, `tests.md` (docs), `lora.png` (classification fixture)
- `test_images/`, `img/` — sample images and README screenshots
- `utils/` — standalone scripts (see Gotchas)
- `dataset_tools/` — dataset v3 pipeline: signal generators, HackRF/RTL-SDR/SDR++ capture, curation, training, evaluation; `manifest.csv` and `splits.csv` (group-held-out split). Scratch output goes to git-ignored `tmp/dataset/`
- `docs/` — `DATASET_GUIDE.md` (how the dataset was built), `DATASET_V3.md` (dataset + model summary), `APP_EVALUATION.md`
- `.github/copilot-instructions.md` — overlaps this file; keep both consistent

## Architecture

- **Singletons at import time:** `CLIENT` (Chroma `PersistentClient("classidyne_db")`) and `EXTRACTOR` (`RadioNetExtractor`). Device is chosen cuda → mps → cpu. The architecture and preprocessing come from the checkpoint; `CLASSIDYNE_MODEL` overrides the checkpoint path. Paths (`./RadioNet/RadioNet.pth`, `datasets/...`, `known_frequencies.json`, `static`) are relative to the CWD, so run from the repo root.
- **Two collections:** `waterfall` and `fft`, both created with cosine space (`hnsw:space`). Constants: `VALID_COLLECTIONS`, `WATERFALL_PATH`, `FFT_PATH`, `ACCEPTED_FILETYPES` (jpeg/jpg/png/gif/tiff/tif/bmp/webp).
- **Embedding pipeline:** `embed_all_datasets()` runs as a FastAPI background task (`/api/start-embedding`) and walks waterfall then FFT. Each file is SHA-256 hashed; the hash is the Chroma ID and the duplicate key (not filename). Known hashes are fetched once up front; inserts are batched at 500. Metadata per item: `filepath`, `filehash`, `class` (= parent directory name). Global `embedding_status` (`EmbeddingStatus` enum) tracks progress; a new run is only accepted when `Idle` or `Fatal Error`.
- **Classification:** `/api/classify` embeds the upload, queries top 20, converts cosine distance to similarity (`1 - distance`), keeps results ≥ `similarity_threshold` (default 0.5), tallies class counts into confidence percentages, attaches frequency ranges, and returns a base64 collage (5×4 grid of 150px tiles, decoded in parallel by `TILE_POOL`). The collage reads source images from `filepath`, so deleting dataset files breaks it (logged, skipped).
- **Frequency data:** `known_frequencies.json` only — not in or derived from the vector DB. `/api/identify_frequency?freq=` takes Hz.
- **Endpoints:** `POST /api/classify`, `POST /api/start-embedding`, `GET /api/stats`, `GET /api/find_image`, `DELETE /api/delete_image`, `GET /api/waterfall_types`, `GET /api/fft_types`, `GET /api/type_collage`, `GET /api/identify_frequency`. CORS is wide open (`*`).
- **Frontend:** React 19 + MUI 7 (dark theme) + React Router 7 + React Query 5 + `react-easy-crop`, TypeScript, Create React App. Calls backend via relative `/api/...` fetches; no typed API client.
- **Static serving:** `app.py` mounts `static/` at `/` last (after API routes). Frontend changes do not appear until built and copied into `static/`.

## Key Conventions

- **Never commit.** The maintainer commits manually. When work is done, suggest a commit message and a `git add` command instead.

- All API responses use `{"success": bool, "message": str, ...}`, even on errors with HTTP status codes (400 bad input, 404 not found, 409 ambiguous, 500 failure). Preserve this.
- Endpoints taking `collection` must validate against `VALID_COLLECTIONS` (guards invalid queries and path traversal).
- Image lookup/delete (`find_image`, `delete_image`) share `_resolve_identifier_candidates` with strict priority: exact Chroma ID/hash → exact filepath → partial basename match. Multiple matches on delete return 409 with a `matches` list.
- Classification and embedding preprocessing convert to grayscale, and for v3 checkpoints squash the **whole** frame to 224×224 (640 px LANCZOS thumbnail, then bicubic resize — identical to `dataset_tools/train/common.py`), then L2-normalize embeddings. Keep query, indexed and training preprocessing identical; changing the model or preprocessing means deleting `classidyne_db/` and re-embedding.
- Deleting via the API removes the Chroma entry only; the file stays in `datasets/`, and re-embedding will re-add it. To reset the DB, delete `classidyne_db/` and re-embed.
- Existing code uses `# FIX:` comments for past bug-fix rationale; keep comments sparse and match surrounding style.

## Tests

- Backend tests are integration tests over HTTP against a live server — start `python app.py` first (embedded dataset required for meaningful classification results).
- `tests/conftest.py` provides the `client` fixture (`httpx.Client`, 30 s timeout, base URL `http://localhost:${CLASSIDYNE_PORT:-5000}`). Update it if the backend runs elsewhere.
- Covered: stats, waterfall/fft types, classify (waterfall & fft, using `tests/lora.png`), find_image, identify_frequency, type_collage. Not covered: delete_image, start-embedding.
- Frontend: `src/App.test.tsx` is the unmodified CRA stub and is not a trustworthy product test; Jest currently fails resolving `react-router-dom` from `App.tsx`.

## Gotchas

- Venv naming is inconsistent: setup docs use `venv-classidyne`, while `tests/tests.md` references `venv`. Either works; just activate one that has `requirements.txt` installed.
- `utils/transform.py` **renames every file** under `./datasets` to its MD5 hash, and `utils/remove-corrupted-images.py` **deletes** corrupted images. Both run on import/execute with hardcoded paths and no confirmation — don't run them casually. Renaming changes `filepath`, so re-embed afterward.
- `requirements.txt` is unpinned and omits `torchvision` (imported by `app.py`; normally pulled in alongside `timm`/`torch` installs — verify if imports fail). It also includes `httpx` for tests.
