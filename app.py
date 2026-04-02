import json
import torch
from PIL import Image, ImageDraw
import timm
from sklearn.preprocessing import normalize
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import chromadb
import hashlib
from collections import Counter
from enum import Enum
import base64
from torchvision import transforms as tv_transforms
from fastapi import FastAPI, UploadFile, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import io
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import uvicorn
import socket
import sys


logger = logging.getLogger("uvicorn")


# ----------------- Embedding Status Enum -----------------

class EmbeddingStatus(str, Enum):
    IDLE = "Idle"
    PROCESSING_WATERFALL = "Processing Waterfall Images"
    PROCESSING_FFT = "Processing FFT Images"
    FATAL_ERROR = "Fatal Error"


# ----------------- App Constant Helper Functions -----------------

def initialize_database() -> chromadb.ClientAPI:
    try:
        client = chromadb.PersistentClient(path="classidyne_db")
        for collection in ("waterfall", "fft"):
            client.get_or_create_collection(
                name=collection,
                metadata={"hnsw:space": "cosine"}
            )
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize database: {e}")


# ----------------- Feature Extractor -----------------

class RadioNetExtractor:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = timm.create_model(
            "resnet34",
            pretrained=False,
            num_classes=0,      # Removes classification head cleanly — no manual fc deletion needed
            global_pool="avg"
        )

        checkpoint = torch.load("./RadioNet/RadioNet.pth", map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        # Drop any leftover fc/classifier keys defensively
        state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)
        logger.info(f"Model is running on: {self.device}")

        config = resolve_data_config({}, model="resnet34")
        transforms = create_transform(**config)
        if isinstance(transforms, tuple):
            self.preprocess = tv_transforms.Compose(list(transforms))
        else:
            self.preprocess = transforms

    def __call__(self, image):
        if isinstance(image, str):
            input_image = Image.open(image).convert("L").convert("RGB")
        else:
            input_image = image.convert("L").convert("RGB")
        input_image = self.preprocess(input_image)
        if not isinstance(input_image, torch.Tensor):
            input_image = tv_transforms.ToTensor()(input_image)
        input_tensor = input_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        feature_vector = output.squeeze().cpu().numpy()
        normalized_vector = normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
        return normalized_vector


# ----------------- Constants and Global Variables -----------------

ACCEPTED_FILETYPES: tuple = (".jpeg", ".jpg", ".png", ".gif", ".tiff", ".tif", ".bmp", ".webp")
WATERFALL_COLLECTION: str = "waterfall"
FFT_COLLECTION: str = "fft"
WATERFALL_PATH: str = "datasets/waterfall"
FFT_PATH: str = "datasets/fft"
VALID_COLLECTIONS: tuple = (WATERFALL_COLLECTION, FFT_COLLECTION)

CLIENT: chromadb.ClientAPI = initialize_database()
EXTRACTOR: RadioNetExtractor = RadioNetExtractor()

with open("known_frequencies.json", "r") as f:
    KNOWN_FREQUENCIES: dict = json.load(f)

embedding_status: EmbeddingStatus = EmbeddingStatus.IDLE


# ----------------- Pydantic Response Models -----------------

class ClassifyResponse(BaseModel):
    success: bool
    message: str
    class_scores: List[dict]
    collage_image: Optional[str]

class StatsResponse(BaseModel):
    success: bool
    message: str
    embedding_status: str
    waterfall_size: int
    fft_size: int

class GenericResponse(BaseModel):
    success: bool
    message: str


# ----------------- Utility Functions -----------------

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_frequency_range(signal_class):
    def format_frequency(freq):
        if freq is None:
            return "Unknown"
        if freq >= 1e9:
            return f"{freq/1e9:.3f} GHz"
        elif freq >= 1e6:
            return f"{freq/1e6:.3f} MHz"
        else:
            return f"{freq:.0f} Hz"

    freq_range = KNOWN_FREQUENCIES.get(signal_class)
    if freq_range is None:
        return "NA"
    if isinstance(freq_range, list):
        if isinstance(freq_range[0], list):
            ranges = []
            for r in freq_range:
                if r[0] == r[1]:
                    ranges.append(format_frequency(r[0]))
                else:
                    ranges.append(f"{format_frequency(r[0])} - {format_frequency(r[1])}")
            return ", ".join(ranges)
        else:
            if freq_range[0] == freq_range[1]:
                return format_frequency(freq_range[0])
            else:
                return f"{format_frequency(freq_range[0])} - {format_frequency(freq_range[1])}"
    elif isinstance(freq_range, dict):
        ranges = []
        for band, band_range in freq_range.items():
            if isinstance(band_range[0], list):
                band_ranges = []
                for r in band_range:
                    if r[0] == r[1]:
                        band_ranges.append(format_frequency(r[0]))
                    else:
                        band_ranges.append(f"{format_frequency(r[0])} - {format_frequency(r[1])}")
                ranges.append(f"- {band}:  \\n {', '.join(band_ranges)}")
            else:
                if band_range[0] == band_range[1]:
                    ranges.append(f"- {band}:  \\n {format_frequency(band_range[0])}")
                else:
                    ranges.append(f"- {band}:  \\n {format_frequency(band_range[0])} - {format_frequency(band_range[1])}")
        return "  \\n".join(ranges)
    else:
        return "Unknown"


def embed_dataset(folder_path: str, collection_name: str) -> int:
    """
    Embeds all images in folder_path into the named ChromaDB collection.
    Performs a single upfront hash fetch to avoid per-file duplicate queries.
    """
    BATCH_SIZE = 500
    collection = CLIENT.get_collection(collection_name)

    # FIX: Fetch all known hashes once — O(1) set lookup instead of N Milvus queries
    existing = collection.get(include=["metadatas"])
    known_hashes = {m["filehash"] for m in existing["metadatas"] if m and "filehash" in m}

    ids, embeddings, metadatas = [], [], []
    inserted = 0

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if not filename.lower().endswith(ACCEPTED_FILETYPES):
                continue
            try:
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "rb") as file:
                    file_hash = hashlib.sha256(file.read()).hexdigest()

                if file_hash in known_hashes:
                    continue

                embedding = EXTRACTOR(filepath)
                ids.append(file_hash)           # filehash doubles as unique ChromaDB ID
                embeddings.append(embedding.tolist())
                metadatas.append({
                    "filepath": filepath,
                    "filehash": file_hash,
                    "class": os.path.basename(os.path.dirname(filepath))
                })
                known_hashes.add(file_hash)

            except Exception as e:
                logger.warning(f"Error processing file {filename}: {e}")
                continue

            if len(ids) >= BATCH_SIZE:
                collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
                inserted += len(ids)
                ids, embeddings, metadatas = [], [], []

    if ids:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        inserted += len(ids)

    return inserted


def embed_all_datasets():
    global embedding_status
    try:
        embedding_status = EmbeddingStatus.PROCESSING_WATERFALL
        waterfall_inserted = embed_dataset(WATERFALL_PATH, WATERFALL_COLLECTION)
        embedding_status = EmbeddingStatus.PROCESSING_FFT
        fft_inserted = embed_dataset(FFT_PATH, FFT_COLLECTION)
        embedding_status = EmbeddingStatus.IDLE
        logger.info(f"Inserted {waterfall_inserted} waterfall and {fft_inserted} FFT entries")
    except Exception as e:
        logger.error(f"Error embedding datasets: {e}")
        embedding_status = EmbeddingStatus.FATAL_ERROR


def identify_frequency(freq):
    if not isinstance(freq, (int, float)) or freq < 0:
        return ["Invalid frequency"]

    def in_range(min_f, max_f):
        forgiveness = 0.03
        return freq >= min_f * (1 - forgiveness) and freq <= max_f * (1 + forgiveness)

    possible_signals = []
    for signal_type, freq_range in KNOWN_FREQUENCIES.items():
        if freq_range is None:
            continue
        if isinstance(freq_range, list):
            if isinstance(freq_range[0], list):
                for r in freq_range:
                    if in_range(r[0], r[1]):
                        possible_signals.append(signal_type)
                        break
            else:
                if in_range(freq_range[0], freq_range[1]):
                    possible_signals.append(signal_type)
        elif isinstance(freq_range, dict):
            for band, band_range in freq_range.items():
                if isinstance(band_range[0], list):
                    for r in band_range:
                        if in_range(r[0], r[1]):
                            possible_signals.append(signal_type)
                            break
                else:
                    if in_range(band_range[0], band_range[1]):
                        possible_signals.append(signal_type)

    return possible_signals


def _candidate(image_id: str, metadata: dict, collection: str) -> Dict[str, Any]:
    return {
        "id": image_id,
        "metadata": metadata,
        "collection": collection,
    }


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique = {}
    for c in candidates:
        key = (c["collection"], c["id"])
        if key not in unique:
            unique[key] = c
    return list(unique.values())


def _lookup_in_collection(identifier: str, collection_name: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Lookup identifier inside a single collection.
    Priority buckets are returned separately so callers can enforce:
    id > exact filepath > partial filename/path match.
    """
    col = CLIENT.get_collection(collection_name)
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "id_matches": [],
        "path_matches": [],
        "partial_matches": [],
    }

    try:
        res = col.get(ids=[identifier], include=["metadatas"])
        for image_id, meta in zip(res.get("ids", []), res.get("metadatas", [])):
            if meta:
                buckets["id_matches"].append(_candidate(image_id, meta, collection_name))
    except Exception as e:
        logger.warning(f"ID lookup failed for '{identifier}' in {collection_name}: {e}")

    try:
        res = col.get(where={"filepath": {"$eq": identifier}}, include=["metadatas"])
        for image_id, meta in zip(res.get("ids", []), res.get("metadatas", [])):
            if meta:
                buckets["path_matches"].append(_candidate(image_id, meta, collection_name))
    except Exception as e:
        logger.warning(f"Path lookup failed for '{identifier}' in {collection_name}: {e}")

    search_term = os.path.basename(identifier).strip().lower()
    if search_term:
        try:
            all_items = col.get(include=["metadatas"])
            partial = []
            for image_id, meta in zip(all_items.get("ids", []), all_items.get("metadatas", [])):
                if not meta:
                    continue
                filepath = meta.get("filepath", "")
                basename = os.path.basename(filepath)
                if search_term in basename.lower() or search_term in filepath.lower():
                    partial.append(_candidate(image_id, meta, collection_name))

            partial.sort(key=lambda c: c["metadata"].get("filepath", ""))
            buckets["partial_matches"] = partial
        except Exception as e:
            logger.warning(f"Partial lookup failed for '{identifier}' in {collection_name}: {e}")

    buckets["id_matches"] = _dedupe_candidates(buckets["id_matches"])
    buckets["path_matches"] = _dedupe_candidates(buckets["path_matches"])
    buckets["partial_matches"] = _dedupe_candidates(buckets["partial_matches"])
    return buckets


def _resolve_identifier_candidates(identifier: str) -> List[Dict[str, Any]]:
    id_matches, path_matches, partial_matches = [], [], []
    for col_name in VALID_COLLECTIONS:
        buckets = _lookup_in_collection(identifier, col_name)
        id_matches.extend(buckets["id_matches"])
        path_matches.extend(buckets["path_matches"])
        partial_matches.extend(buckets["partial_matches"])

    if id_matches:
        return _dedupe_candidates(id_matches)
    if path_matches:
        return _dedupe_candidates(path_matches)
    return _dedupe_candidates(partial_matches)


# ----------------- FastAPI Setup -----------------

app = FastAPI(
    title="Classidyne API",
    description="APIs for searching, classifying, and managing signal images using deep learning and ChromaDB.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to known origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- API Endpoints -----------------

@app.post(
    "/api/classify",
    summary="Classify an uploaded image",
    response_description="Classification results and collage image",
    response_model=ClassifyResponse
)
def classify(
    query_image: UploadFile = Form(..., description="Image file to classify"),
    collection: str = Form(..., description="Collection to search in (waterfall or fft)"),
    similarity_threshold: float = Form(0.5, description="Minimum similarity score (0-1) to consider a match")
):
    # FIX: Validate collection parameter to prevent invalid queries / path traversal
    if collection not in VALID_COLLECTIONS:
        return JSONResponse(
            {"success": False, "message": f"Invalid collection '{collection}'. Must be one of: {VALID_COLLECTIONS}"},
            status_code=400
        )
    try:
        image = Image.open(query_image.file).convert("L").convert("RGB")
        query_embedding = EXTRACTOR(image).tolist()

        chroma_collection = CLIENT.get_collection(collection)
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=20,
            include=["metadatas", "distances"]
        )

        metadatas = results["metadatas"][0]   # list of metadata dicts
        distances = results["distances"][0]   # cosine distance: 0=identical, 2=opposite

        width, height = 150 * 5, 150 * 4
        concatenated_image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(concatenated_image)
        class_counts = Counter()
        idx = 0

        for meta, distance in zip(metadatas, distances):
            # ChromaDB returns cosine distance; convert to similarity score
            similarity_score = 1.0 - distance
            filepath = meta["filepath"]
            signal_class = meta["class"]

            try:
                img = Image.open(filepath).resize((150, 150))
                if similarity_score >= similarity_threshold and idx < 20:
                    class_counts[signal_class] += 1
                    x = idx % 5
                    y = idx // 5
                    concatenated_image.paste(img, (x * 150, y * 150))
                    draw.text((x * 150, y * 150 + 110), f"Class: {signal_class}", fill=(255, 255, 255))
                    draw.text((x * 150, y * 150 + 125), f"Score: {similarity_score:.2f}", fill=(255, 255, 255))
                    idx += 1
            except FileNotFoundError:
                logger.error(f"File not found: {filepath}\nDid you delete the file from the datasets folder?")

        total_count = sum(class_counts.values())
        collage = None

        if total_count > 0:
            class_scores = sorted([
                {
                    "class": cls,
                    "confidence": count / total_count * 100,
                    "frequency_range": get_frequency_range(cls)
                }
                for cls, count in class_counts.items()
            ], key=lambda x: x["confidence"], reverse=True)
            collage = image_to_base64(concatenated_image)
        else:
            class_scores = []

        return JSONResponse({
            "success": True,
            "message": f"Found {total_count} images with similarity score above {similarity_threshold:.2f}.",
            "class_scores": class_scores,
            "collage_image": collage
        })
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Error classifying image: {e}",
            "class_scores": [],
            "collage_image": None
        }, status_code=500)


@app.post("/api/start-embedding", response_model=GenericResponse)
def start_embedding(background_tasks: BackgroundTasks):
    if embedding_status in (EmbeddingStatus.IDLE, EmbeddingStatus.FATAL_ERROR):
        background_tasks.add_task(embed_all_datasets)
        return {"success": True, "message": "Embedding task started in the background."}
    else:
        return {"success": False, "message": f"Embedding task already running. Status: {embedding_status}"}


@app.get(
    "/api/stats",
    summary="Get embedding status and database sizes",
    response_model=StatsResponse
)
def get_stats():
    try:
        waterfall_count = CLIENT.get_collection("waterfall").count()
        fft_count = CLIENT.get_collection("fft").count()
        return JSONResponse({
            "success": True,
            "message": "Stats fetched successfully.",
            "embedding_status": embedding_status,
            "waterfall_size": waterfall_count,
            "fft_size": fft_count
        })
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return JSONResponse({"success": False, "message": f"Error fetching stats: {e}"}, status_code=500)


@app.delete("/api/delete_image", summary="Delete an image by filehash or filepath")
def delete_image(identifier: str):
    try:
        candidates = _resolve_identifier_candidates(identifier)
        if not candidates:
            return JSONResponse({"success": False, "message": "No images found with the given identifier."}, status_code=404)

        if len(candidates) > 1:
            paths = [c["metadata"].get("filepath", "") for c in candidates[:20]]
            return JSONResponse({
                "success": False,
                "message": "Multiple images match this identifier. Please provide a full file path or file hash.",
                "matches": paths,
            }, status_code=409)

        target = candidates[0]
        col = CLIENT.get_collection(target["collection"])
        col.delete(ids=[target["id"]])
        filepath = target["metadata"].get("filepath", "")
        return JSONResponse({
            "success": True,
            "message": f"Deleted image '{filepath}'.",
            "filepath": filepath,
            "filehash": target["id"],
        })
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        return JSONResponse({"success": False, "message": f"Error deleting image: {e}"}, status_code=500)


@app.get("/api/find_image", summary="Find an image by filehash or filepath")
def find_image(identifier: str):
    try:
        candidates = _resolve_identifier_candidates(identifier)
        if not candidates:
            return JSONResponse({"success": False, "message": "No matching images found."}, status_code=404)

        if len(candidates) > 1:
            paths = [c["metadata"].get("filepath", "") for c in candidates[:20]]
            return JSONResponse({
                "success": False,
                "message": "Multiple images match this identifier. Please provide a full file path or file hash.",
                "matches": paths,
            }, status_code=409)

        found = candidates[0]["metadata"]
        filepath = found["filepath"]
        if found:
            img = Image.open(filepath).resize((150, 150))
            return JSONResponse({
                "success": True,
                "message": "Found a matching image.",
                "filepath": filepath,
                "filehash": found["filehash"],
                "class": found["class"],
                "image": image_to_base64(img)
            })
    except Exception as e:
        logger.error(f"Error finding image: {e}")
        return JSONResponse({"success": False, "message": f"Error finding image: {e}"}, status_code=500)


@app.get("/api/waterfall_types", summary="List all signal types in the waterfall collection")
def get_waterfall_types():
    try:
        base_path = os.path.join("./datasets", "waterfall")
        types = sorted([
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ])
        return JSONResponse({"success": True, "message": "Waterfall types listed successfully.", "types": types})
    except Exception as e:
        logger.error(f"Error listing waterfall types: {e}")
        return JSONResponse({"success": False, "message": f"Error listing waterfall types: {e}"}, status_code=500)


@app.get("/api/fft_types", summary="List all signal types in the FFT collection")
def get_fft_types():
    try:
        base_path = os.path.join("./datasets", "fft")
        types = sorted([
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ])
        return JSONResponse({"success": True, "message": "FFT types listed successfully.", "types": types})
    except Exception as e:
        logger.error(f"Error listing FFT types: {e}")
        return JSONResponse({"success": False, "message": f"Error listing FFT types: {e}"}, status_code=500)


@app.get("/api/type_collage", summary="Get a collage of images for a specific signal type")
def get_type_collage(type: str, collection: str):
    # FIX: Validate collection to prevent path traversal
    if collection not in VALID_COLLECTIONS:
        return JSONResponse(
            {"success": False, "message": f"Invalid collection '{collection}'."},
            status_code=400
        )
    try:
        full_path = os.path.join("./datasets", collection, type)
        img_files = sorted([
            img for img in os.listdir(full_path)
            if img.lower().endswith(ACCEPTED_FILETYPES)
        ])[:25]

        images = [
            Image.open(os.path.join(full_path, img)).resize((150, 150))
            for img in img_files
        ]
        concatenated_image = Image.new("RGB", (150 * 5, 150 * 5))
        for idx, img in enumerate(images):
            x = idx % 5
            y = idx // 5
            concatenated_image.paste(img, (x * 150, y * 150))

        return JSONResponse({
            "success": True,
            "message": "Collage generated successfully.",
            "collage_base64": image_to_base64(concatenated_image)
        })
    except Exception as e:
        logger.error(f"Error generating collage: {e}")
        return JSONResponse({"success": False, "message": f"Error generating collage: {e}"}, status_code=500)


@app.get("/api/identify_frequency", summary="Identify possible signal types for a given frequency in Hz")
def api_identify_frequency(freq: float):
    """
    Returns a list of signal types whose known frequency ranges include the given frequency.
    freq should be provided in Hz (e.g. 2.4e9 for 2.4 GHz).
    """
    signals = identify_frequency(freq)
    if signals == ["Invalid frequency"]:
        return JSONResponse({"success": False, "message": "Invalid frequency provided.", "signals": []}, status_code=400)
    return JSONResponse({
        "success": True,
        "message": f"Found {len(signals)} matching signal type(s).",
        "signals": signals
    })


app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    HOST = "0.0.0.0"
    START_PORT = 5000
    END_PORT = 5005

    for port in range(START_PORT, END_PORT + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, port))
                s.close()
                print(f"Starting server on port {port}")
                uvicorn.run("app:app", host=HOST, port=port, reload=False)
                break
            except OSError:
                print(f"Port {port} unavailable, trying next...")
    else:
        print(f"No ports available between {START_PORT} and {END_PORT}. Server aborting.")
        sys.exit(1)
