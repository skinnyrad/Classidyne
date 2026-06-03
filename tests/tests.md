# Test Suite Documentation

## Running the Tests

Run the test suite using the project's virtual environment:

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

The tests connect to the running Classidyne server. The backend defaults to port 5000. To override the port when running via `pytest`, set the `CLASSIDYNE_PORT` environment variable:

```bash
CLASSIDYNE_PORT=5005 python -m pytest tests/ -v
```

## Test Files

### `conftest.py`

**Purpose:** Provides Pytest fixtures for the test suite.

**Key Components:**
- `client` fixture: Creates an `httpx.Client` configured to connect to the Classidyne API at `http://localhost:<port>` (defaults to 5000, or `CLASSIDYNE_PORT` environment variable) with a 30-second timeout.
- This fixture is automatically injected into all test functions in `test_api.py` that accept a `client` parameter.

### `test_api.py`

**Purpose:** Integration tests for the Classidyne API endpoints.

**Test Functions:**

| Test Function | API Endpoint | Purpose |
|---------------|--------------|---------|
| `test_stats` | `GET /api/stats` | Verifies the API returns expected statistics (embedding status, collection sizes) |
| `test_waterfall_types` | `GET /api/waterfall_types` | Verifies waterfall signal types are returned (e.g., "lora") |
| `test_fft_types` | `GET /api/fft_types` | Verifies FFT signal types are returned |
| `test_classify_waterfall` | `POST /api/classify` (waterfall) | Tests classification using the waterfall collection; expects "lora" as top result |
| `test_classify_fft` | `POST /api/classify` (fft) | Tests classification using the FFT collection; expects no matches for a waterfall image |
| `test_find_image` | `GET /api/find_image` | Tests image lookup by identifier; expects multiple matches |
| `test_identify_frequency` | `GET /api/identify_frequency` | Tests frequency identification (915 MHz); expects "lora" in results |
| `test_type_collage` | `GET /api/type_collage` | Tests collage generation for a signal type; expects base64 collage data |

**Test Image:**
- Uses `tests/lora.png` for classification tests.

**Standalone Execution:**
The script can also run independently using argparse to specify host/port:
```bash
python tests/test_api.py --host localhost --port 5000
```
