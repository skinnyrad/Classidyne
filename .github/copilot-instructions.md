# Classidyne Copilot Instructions

## Build, test, and run commands

- **Backend install:** `python3 -m venv venv-classidyne && source venv-classidyne/bin/activate && pip install -r requirements.txt`
- **Backend run:** `python app.py`
- **Frontend install:** `cd frontend && npm ci`
- **Frontend dev server:** `cd frontend && npm start`
- **Frontend production build:** `cd frontend && npm run build`
- **Frontend full test run:** `cd frontend && CI=true npm test -- --runInBand --watch=false`
- **Frontend single test file:** `cd frontend && CI=true npm test -- --runInBand src/App.test.tsx --watch=false`
- **Linting:** there is no standalone lint script; Create React App runs ESLint during `cd frontend && npm run build` and `cd frontend && npm start`

## High-level architecture

- The backend is a single FastAPI app in `app.py`. It owns model loading, ChromaDB access, embedding jobs, image lookup/delete APIs, and the classification API.
- `RadioNetExtractor` loads `./RadioNet/RadioNet.pth` at import time and is kept as a module-level singleton. `CLIENT` is also initialized once as a persistent Chroma client rooted at `classidyne_db/`.
- The vector store is split into two collections, `waterfall` and `fft`. Most backend behavior mirrors that split: embedding, stats, type listing, type collage, and classification all work against one or both of those collections.
- Dataset ingestion is filesystem-driven. The backend walks `datasets/waterfall/<signal-type>/` and `datasets/fft/<signal-type>/`, hashes each image, skips duplicates by hash, extracts embeddings, and writes metadata containing `filepath`, `filehash`, and `class`.
- Frequency metadata is not inferred from embeddings; it comes from `known_frequencies.json` and is attached when returning classification results.
- The editable React app lives in `frontend/src/` and uses React Router + React Query + MUI. It talks to the backend with relative `/api/...` fetches rather than a typed client.
- `python app.py` serves the checked-in `static/` bundle via `StaticFiles`. `cd frontend && npm run build` writes to `frontend/build`, so frontend changes do **not** automatically update the UI served by FastAPI unless the built assets are synced into `static/`.

## Key conventions

- Git LFS is required for the model file. If `RadioNet/RadioNet.pth` was not pulled correctly, backend import/startup will fail before the API serves requests.
- The app expects the first real workflow to be embedding the dataset into ChromaDB. A fresh checkout is not usable for classification until the image database has been built.
- Backend responses consistently return JSON objects with `success` and `message`, even for many error cases that also set HTTP status codes. Preserve that response shape when extending endpoints.
- Image management resolves identifiers with a strict priority: exact Chroma ID/file hash first, exact filepath second, partial basename/path match last. `find_image` and `delete_image` share that behavior.
- Classification works on grayscale-converted images but still feeds the model RGB tensors (`convert("L").convert("RGB")`). Keep preprocessing aligned with that flow.
- The duplicate check during embedding is hash-based, not filename-based. Utility scripts in `utils/` also assume the dataset tree under `./datasets` is the source of truth.
- The current frontend test harness is still in CRA/Jest form and the checked-in `src/App.test.tsx` is not a trustworthy product test; test runs currently fail in Jest while resolving `react-router-dom` from `App.tsx`.
