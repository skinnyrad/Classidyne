# Classidyne API Documentation

The Classidyne API provides endpoints for signal classification, database management, and frequency identification.

## Base URL

By default, the backend runs on `http://localhost:5000` (or the first available port up to 5005).

## Response Format

All responses follow a consistent JSON structure:

```json
{
  "success": boolean,
  "message": "Description of the result or error",
  ... (additional fields)
}

```

---

## Classification Endpoints

### Post Classify

`POST /api/classify`

Classifies an uploaded signal image by comparing its embedding against the specified collection in ChromaDB.

**Request (Multipart Form Data):**

* `query_image`: (File) The image to classify.
* `collection`: (String) Either `waterfall` or `fft`.
* `similarity_threshold`: (Float, optional, default: 0.5) Minimum similarity score (0.0 to 1.0) to consider a match.

**Example `curl`:**

```bash
curl -X POST http://localhost:5000/api/classify \
  -F "query_image=@/path/to/your/signal.png" \
  -F "collection=waterfall" \
  -F "similarity_threshold=0.5"

```

**Example Python:**

```python
import requests

url = "http://localhost:5000/api/classify"
files = {
    "query_image": open("/path/to/your/signal.png", "rb")
}
data = {
    "collection": "waterfall",
    "similarity_threshold": 0.5
}

response = requests.post(url, files=files, data=data)
print(response.json())

```

**Response:**

```json
{
  "success": true,
  "message": "Found X images with similarity score above Y.",
  "class_scores": [
    {
      "class": "wifi",
      "confidence": 85.5,
      "frequency_range": "2.412 GHz - 2.484 GHz, 5.170 GHz - 5.835 GHz"
    }
  ],
  "collage_image": "base64_encoded_png_string"
}
```

> **Note:** `collage_image` is base64 PNG data and may be `null` when no matching images are found.

---

## Database Management

### Get Stats

`GET /api/stats`

Returns the current status of the embedding process and the number of items in each collection.

**Example `curl`:**

```bash
curl http://localhost:5000/api/stats

```

**Example Python:**

```python
import requests

response = requests.get("http://localhost:5000/api/stats")
print(response.json())

```

**Response:**

```json
{
  "success": true,
  "message": "Stats fetched successfully.",
  "embedding_status": "Idle",
  "waterfall_size": 1250,
  "fft_size": 800
}

```

> **Note:** `embedding_status` can be: `Idle`, `Processing Waterfall Images`, `Processing FFT Images`, or `Fatal Error`.

### Start Embedding

`POST /api/start-embedding`

Triggers a background task to walk the `datasets/` directory and extract embeddings for any new images.

**Example `curl`:**

```bash
curl -X POST http://localhost:5000/api/start-embedding

```

**Example Python:**

```python
import requests

response = requests.post("http://localhost:5000/api/start-embedding")
print(response.json())

```

**Response:**

```json
{
  "success": true,
  "message": "Embedding task started in the background."
}

```

### Find Image

`GET /api/find_image`

Finds an image by its file hash, exact file path, or partial filename.

**Parameters:**

* `identifier`: (String) File hash, absolute path, or filename.

**Example `curl`:**

```bash
curl "http://localhost:5000/api/find_image?identifier=abc123hash"

```

**Example Python:**

```python
import requests

params = {"identifier": "abc123hash"}
response = requests.get("http://localhost:5000/api/find_image", params=params)
print(response.json())

```

**Response:**

```json
{
  "success": true,
  "message": "Found a matching image.",
  "filepath": "datasets/waterfall/wifi/sample.png",
  "filehash": "abc123hash",
  "class": "wifi",
  "image": "base64_encoded_thumbnail"
}

```

> **Note:** If the identifier matches multiple images, the API returns HTTP 409 with a `matches` list. If no image matches, it returns HTTP 404.

### Delete Image

`DELETE /api/delete_image`

Removes an image's embedding from the vector database. Note: This does **not** delete the file from the filesystem.

**Parameters:**

* `identifier`: (String) File hash or exact file path.

**Example `curl`:**

```bash
curl -X DELETE "http://localhost:5000/api/delete_image?identifier=abc123hash"

```

**Example Python:**

```python
import requests

params = {"identifier": "abc123hash"}
response = requests.delete("http://localhost:5000/api/delete_image", params=params)
print(response.json())

```

**Response:**

```json
{
  "success": true,
  "message": "Deleted image '...'.",
  "filepath": "...",
  "filehash": "..."
}

```

> **Note:** If the identifier matches multiple images, the API returns HTTP 409 with a `matches` list asking for a more specific identifier.

---

## Metadata & Discovery

### List Waterfall Types

`GET /api/waterfall_types`

Returns a list of all signal classes currently present in the `datasets/waterfall` directory.

**Example `curl`:**

```bash
curl http://localhost:5000/api/waterfall_types

```

**Example Python:**

```python
import requests

response = requests.get("http://localhost:5000/api/waterfall_types")
print(response.json())

```

### List FFT Types

`GET /api/fft_types`

Returns a list of all signal classes currently present in the `datasets/fft` directory.

**Example `curl`:**

```bash
curl http://localhost:5000/api/fft_types

```

**Example Python:**

```python
import requests

response = requests.get("http://localhost:5000/api/fft_types")
print(response.json())

```

### Get Type Collage

`GET /api/type_collage`

Generates a 5x5 collage of example images for a specific signal type.

**Parameters:**

* `type`: (String) The signal class name (e.g., `wifi`).
* `collection`: (String) `waterfall` or `fft`.

**Example `curl`:**

```bash
curl "http://localhost:5000/api/type_collage?type=wifi&collection=waterfall"

```

**Example Python:**

```python
import requests

params = {"type": "wifi", "collection": "waterfall"}
response = requests.get("http://localhost:5000/api/type_collage", params=params)
print(response.json())

```

**Response:**

```json
{
  "success": true,
  "message": "Collage generated successfully.",
  "collage_base64": "base64_encoded_png_string"
}
```

> **Note:** The response field for the collage is `collage_base64`. The API generates up to 25 images in a 5×5 collage.

---

## Frequency Identification

### Identify Frequency

`GET /api/identify_frequency`

Returns a list of potential signal types that are known to operate at the given frequency.

**Parameters:**

* `freq`: (Float) Frequency in Hertz (e.g., `2400000000` for 2.4 GHz).

**Example `curl`:**

```bash
curl "http://localhost:5000/api/identify_frequency?freq=2400000000"

```

**Example Python:**

```python
import requests

params = {"freq": 2400000000}
response = requests.get("http://localhost:5000/api/identify_frequency", params=params)
print(response.json())

```

**Response:**

```json
{
  "success": true,
  "message": "Found 2 matching signal type(s).",
  "signals": ["wifi", "bluetooth"]
}
```

> **Note:** If an invalid frequency is provided, the API responds with HTTP 400, `success: false`, and `signals: []`.
