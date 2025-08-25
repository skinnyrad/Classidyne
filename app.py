import json
import torch
from PIL import Image, ImageDraw
import timm
from sklearn.preprocessing import normalize
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
from pymilvus import MilvusClient
import hashlib
from collections import Counter
from time import sleep
import base64
from torchvision import transforms as tv_transforms
from fastapi import FastAPI, UploadFile, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import io
from typing import List
import logging

logger = logging.getLogger("uvicorn")

# ----------------- App Constant Helper Functions -------------

def initialize_database() -> MilvusClient:
    try:
        client = MilvusClient("classidyne.db")
        collections = ["waterfall", "fft"]
        for collection in collections:
            if not client.has_collection(collection_name=collection):
                client.create_collection(
                    collection_name=collection,
                    dimension=512,
                    auto_id=True
                )
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize database: {e}")

# Feature Extractor class
class RadioNetExtractor:
    def __init__(self):
        # First create model with num_classes matching training
        self.model = timm.create_model("resnet34", 
                                     pretrained=False,
                                     num_classes=0)  
        
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load('./RadioNet/RadioNet.pth', map_location=device)
        
        # Remove unexpected keys from state dict
        state_dict = checkpoint['model_state_dict']
        for key in list(state_dict.keys()):
            if 'fc' in key:  # Remove classification head weights
                del state_dict[key]
        
        # Load the modified state dict
        self.model.load_state_dict(state_dict, strict=False)
              
        # Rest of your initialization code
        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        logger.info(f"Model is running on: {self.device}")
        
        config = resolve_data_config({}, model="resnet34")
        transforms = create_transform(**config)
        # If create_transform returns a tuple, compose them
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
    
# ------------- Constants and global variables -------------    

ACCEPTED_FILETYPES: tuple = (".jpeg", ".jpg", ".png", ".gif", ".tiff", ".tif", ".bmp", ".webp")
WATERFALL_COLLECTION: str = "waterfall"
FFT_COLLECTION: str = "fft"
WATERFALL_PATH: str = "datasets/waterfall"
FFT_PATH: str = "datasets/fft"
CLIENT: MilvusClient = initialize_database()
EXTRACTOR: RadioNetExtractor = RadioNetExtractor()
with open('known_frequencies.json', 'r') as f:
    KNOWN_FREQUENCIES: dict = json.load(f)
    
embedding_status: str = "Idle"
    
# ----------------- Utility Functions -----------------    
    
def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def insert_batch(collection, batch_data) -> int:
    try:
        insert_result = CLIENT.insert(collection, batch_data)
        return insert_result.get('insert_count', 0)
    except Exception as e:
        logger.error(f"Error inserting batch: {e}")
        return 0

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
            # Multiple ranges or single frequencies
            ranges = []
            for r in freq_range:
                if r[0] == r[1]:  # Single frequency
                    ranges.append(format_frequency(r[0]))
                else:  # Range
                    ranges.append(f"{format_frequency(r[0])} - {format_frequency(r[1])}")
            return ", ".join(ranges)
        else:
            # Single range or single frequency
            if freq_range[0] == freq_range[1]:  # Single frequency
                return format_frequency(freq_range[0])
            else:  # Range
                return f"{format_frequency(freq_range[0])} - {format_frequency(freq_range[1])}"
    elif isinstance(freq_range, dict):
        # For sub bands like cellular and WiFi
        ranges = []
        for band, band_range in freq_range.items():
            if isinstance(band_range[0], list):
                band_ranges = []
                for r in band_range:
                    if r[0] == r[1]:  # Single frequency
                        band_ranges.append(format_frequency(r[0]))
                    else:  # Range
                        band_ranges.append(f"{format_frequency(r[0])} - {format_frequency(r[1])}")
                ranges.append(f"- {band}:  \n {', '.join(band_ranges)}")
            else:
                if band_range[0] == band_range[1]:  # Single frequency
                    ranges.append(f"- {band}:  \n {format_frequency(band_range[0])}")
                else:  # Range
                    ranges.append(f"- {band}:  \n {format_frequency(band_range[0])} - {format_frequency(band_range[1])}")
        return "  \n".join(ranges)
    else:
        return "Unknown"
    
def embed_dataset(folder_path, collection):
    BATCH_SIZE = 500
    batch_data = []
    inserted = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(ACCEPTED_FILETYPES):
                try:
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, "rb") as file:
                        file_hash = hashlib.sha256(file.read()).hexdigest()

                    already_exists = CLIENT.query(
                        collection,
                        filter=f'filehash == "{file_hash}"',
                        output_fields=["filehash"],
                        limit=1
                    )
                    if already_exists: continue

                    image_embedding = EXTRACTOR(filepath)
                    batch_data.append({
                        "vector": image_embedding.tolist(),
                        "filepath": filepath,
                        "filehash": file_hash,
                        "class": os.path.basename(os.path.dirname(filepath))
                    })
                except Exception as e:
                    logger.warning(f"Error processing file {filename}: {e}")
                    continue

                if len(batch_data) >= BATCH_SIZE:
                    inserted += insert_batch(collection, batch_data)
                    batch_data = []
    if batch_data:
        inserted += insert_batch(collection, batch_data)
    return inserted

def embed_all_datasets():
    global embedding_status
    try:
        embedding_status = "Processing Waterfall Images"
        waterfall_inserted = embed_dataset(WATERFALL_PATH, WATERFALL_COLLECTION)
        embedding_status = "Processing FFT Images"
        fft_inserted = embed_dataset(FFT_PATH, FFT_COLLECTION)
        embedding_status = "Idle"
        logger.info(f"Inserted {waterfall_inserted} waterfall and {fft_inserted} fft entries")
    except Exception as e:
        logger.error(f"Error embedding datasets: {e}")
        embedding_status = "Fatal Error"
    
# ----------------- FastAPI Setup -----------------

app = FastAPI(
    title="Classidyne API",
    description="APIs for searching, classifying, and managing signal images using deep learning and Milvus vector DB.",
    version="1.0.0"
)

# CONFIGure CORS for future React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- API Endpoints -----------------

@app.post("/api/classify", summary="Classify an uploaded image", response_description="Classification results and collage image")
def classify(
    query_image: UploadFile = Form(..., description="Image file to classify"),
    collection: str = Form(..., description="Collection to search in (waterfall or fft)"),
    similarity_threshold: float = Form(0.5, description="Minimum similarity score to consider a match")
    ):
    try:
        image = Image.open(query_image.file).convert("L").convert("RGB")
        results = CLIENT.search(
            collection,
            data=[EXTRACTOR(image)],
            output_fields=["filepath", "filehash", "class"],
            params={"metric_type": "COSINE"},
            limit=20
        )
        
        width, height = 150 * 5, 150 * 4
        concatenated_image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(concatenated_image)
        class_counts = Counter()
        idx = 0

        # Process all in one loop, including class count tally
        for result in results:
            for hit in result:
                filepath = hit["entity"]["filepath"]
                signal_class = hit["entity"]["class"]
                similarity_score = hit["distance"]
                try:
                    img = Image.open(filepath)
                    img = img.resize((150, 150))
                    if similarity_score >= similarity_threshold and idx < 20:
                        class_counts[signal_class] += 1  # Compute class tally within this loop
                        x = idx % 5
                        y = idx // 5
                        concatenated_image.paste(img, (x * 150, y * 150))
                        draw.text((x * 150, y * 150 + 110), f"Class: {signal_class}", fill=(255, 255, 255))
                        draw.text((x * 150, y * 150 + 125), f"Score: {similarity_score:.2f}", fill=(255, 255, 255))
                        idx += 1
                except FileNotFoundError:
                    logger.error(f"File not found: {filepath}\nDid you delete the file from the datasets folder?")

        # Now compute class scores directly from the tally
        class_scores = []
        total_count = sum(class_counts.values())
        if total_count > 0:
            for cls, count in class_counts.items():
                class_scores.append({
                    "class": cls,
                    "confidence": count / total_count * 100,
                    "frequency_range": get_frequency_range(cls)
                })
        else:
            concatenated_image = None  # No images met threshold

        # Sort class_scores by descending confidence
        class_scores = sorted(class_scores, key=lambda x: x["confidence"], reverse=True)
        return JSONResponse({
            "success": True,
            "message": f"Found {total_count} images with similarity score above {similarity_threshold:.2f}.",
            "class_scores": class_scores,
            "collage_image": image_to_base64(concatenated_image) if concatenated_image else None
        })
    except Exception as e:
        logger.error(f"Error querying image: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Error querying image: {e}",
            "class_scores": [],
            "collage_image": None
        }, status_code=500)
        
@app.post("/api/start-embedding")
def start_embedding(background_tasks: BackgroundTasks):
    if embedding_status == "Idle" or embedding_status == "Finished" or embedding_status == "Fatal Error":
        background_tasks.add_task(embed_all_datasets)
        return {"success": True, "message": "Embedding task started in the background."}
    else:
        return {"success": False, "message": f"Embedding task is already running. Current status: {embedding_status}"}
    
@app.get("/api/stats", summary="Get the current status of the embedding process and database sizes")
def get_stats():
    try:
        return JSONResponse({
            "success": True,
            "message": "Stats fetched successfully.",
            "embedding_status": embedding_status,
            "waterfall_size": CLIENT.get_collection_stats(collection_name="waterfall").get('row_count', 0),
            "fft_size": CLIENT.get_collection_stats(collection_name="fft").get('row_count', 0)
        })
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return JSONResponse({"success": False, "message": f"Error fetching stats: {e}"}, status_code=500)
        
@app.delete("/api/delete_image", summary="Attempt to delete an image by its identifier (filehash or filename)")
def delete_image(identifier):
    try:
        # Delete by filehash
        deleted = CLIENT.delete(collection_name="waterfall", filter=f'hash == "{identifier}"').get('deleted_count', 0)
        deleted += CLIENT.delete(collection_name="fft", filter=f'hash == "{identifier}"').get('deleted_count', 0)
        
        # If identifier is a filepath, strip to filename
        filename = os.path.basename(identifier)
        # Delete by filename (should match filepaths ending with filename)
        deleted += CLIENT.delete(collection_name="waterfall", filter=f'filepath like "%/{filename}"').get('deleted_count', 0)
        deleted += CLIENT.delete(collection_name="fft", filter=f'filepath like "%/{filename}"').get('deleted_count', 0)
        
        if deleted:
            return JSONResponse({"success": True, "message": f"Deleted {deleted} images with identifier '{identifier}'."})
        else:
            return JSONResponse({"success": False, "message": "No images found with the given identifier."}, status_code=404)
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        return JSONResponse({"success": False, "message": f"Error deleting image: {e}"}, status_code=500)


@app.get("/api/find_image", summary="Find an image by its identifier (filehash or filename)")
def find_image(identifier):
    try:
        # Query both collections by hash
        results = CLIENT.query(collection_name="waterfall", filter=f'hash == "{identifier}"', limit=1)
        if not results or len(results) == 0:
            results = CLIENT.query(collection_name="fft", filter=f'hash == "{identifier}"', limit=1)
        
        if not results or len(results) == 0:
            # If identifier is a filepath, strip to filename
            filename = os.path.basename(identifier)
            # Try pattern matching for filepaths ending with filename
            results = CLIENT.query(collection_name="waterfall", filter=f'filepath like "%/{filename}"', limit=1)
        if not results or len(results) == 0:
            results = CLIENT.query(collection_name="fft", filter=f'filepath like "%/{filename}"', limit=1)
                
        if results and results[0]:
            logger.info(f"Found a matching image.")
            hit = results[0]
            filepath = hit['filepath']
            signal_class = hit['class']
            filehash = hit['filehash']
            
            # Open and resize the image
            img = Image.open(filepath)
            img = img.resize((150, 150))
            
            return JSONResponse({
                "success": True,
                "message": f"Found a matching image.",
                "filepath": filepath,
                "filehash": filehash,
                "class": signal_class,
                "image": image_to_base64(img)
            })
                
        else: 
            logger.info("No matching images found.")
            return JSONResponse({"success": False, "message": "No matching images found."}, status_code=404)
    except Exception as e:
        logger.error(f"Error querying image: {e}")
        return JSONResponse({"success": False, "message": f"Error querying image: {e}"}, status_code=500)
    
@app.get('/api/waterfall_types', summary="List all signal types in the waterfall collection")
def get_waterfall_types():
    try:
        base_path = os.path.join('./datasets', 'waterfall')
        types = sorted([
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ])
        return JSONResponse({
            "success": True,
            "message": "Waterfall types listed successfully.",
            "types": types
        })
    except Exception as e:
        logger.error(f"Error listing waterfall types: {e}")
        return JSONResponse(
            {"success": False, "message": f"Error listing waterfall types: {e}"},
            status_code=500
        )

@app.get('/api/fft_types', summary="List all signal types in the fft collection")
def get_fft_types():
    try:
        base_path = os.path.join('./datasets', 'fft')
        types = sorted([
            name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ])
        return JSONResponse({
            "success": True,
            "message": "FFT types listed successfully.",
            "types": types
        })
    except Exception as e:
        logger.error(f"Error listing FFT types: {e}")
        return JSONResponse(
            {"success": False, "message": f"Error listing FFT types: {e}"},
            status_code=500
        )

@app.get('/api/type_collage', summary="Get a collage of images for a specific signal type")
def get_type_collage(type: str, collection: str):
    try:
        # Prepare full directory path
        full_path = os.path.join('./datasets', collection, type)
        # Gather and resize images
        # Step 1: Get only the filenames with accepted extensions, limited to 25
        img_files = [
            img for img in os.listdir(full_path)
            if img.lower().endswith(ACCEPTED_FILETYPES)
        ][:25]

        # Step 2: Open and resize only those 25 files
        images = [
            Image.open(os.path.join(full_path, img)).resize((150, 150))
            for img in img_files
        ]
        # Create a new blank (RGB) image, 5x5 grid of 150x150 squares
        concatenated_image = Image.new("RGB", (150 * 5, 150 * 5))
        # Paste images into the grid
        for idx, img in enumerate(images):
            x = idx % 5
            y = idx // 5
            concatenated_image.paste(img, (x * 150, y * 150))
        
        concatenated_image_base64 = image_to_base64(concatenated_image)
        return JSONResponse({
            "success": True,
            "message": "Collage generated successfully.",
            "collage_base64": concatenated_image_base64
        })
    except Exception as e:
        logger.error(f"Error generating collage: {e}")
        return JSONResponse(
            {"success": False, "message": f"Error generating collage: {e}"},
            status_code=500
        )

def identify_frequency(freq):
    # Check if freq is a valid number
    if not isinstance(freq, (int, float)) or freq < 0:
        return ["Invalid frequency"]
    
    def in_range(min, max):
        forgiveness = 0.03
        return freq >= min * (1-forgiveness) and freq <= max * (1+forgiveness)

    possible_signals = []
    for signal_type, freq_range in KNOWN_FREQUENCIES.items():
        if freq_range is None:
            continue # not found at a specific frequency
        if isinstance(freq_range, list):
            if isinstance(freq_range[0], list):
                # Multiple ranges or single frequencies
                for r in freq_range:
                    if in_range(r[0],r[1]):
                        possible_signals.append(signal_type)
                        break
            else:
                # Single range or single frequency
                if in_range(freq_range[0],freq_range[1]):
                    possible_signals.append(signal_type)
    
        elif isinstance(freq_range, dict):
            # For sub bands like cellular and WiFi
            for band, band_range in freq_range.items():
                if isinstance(band_range[0], list):
                    for r in band_range:
                        if in_range(r[0],r[1]): 
                            possible_signals.append(signal_type)
                            break
                else:
                    if in_range(band_range[0], band_range[1]):  # Single frequency
                        possible_signals.append(signal_type)
                    
    return possible_signals

app.mount("/", StaticFiles(directory="static", html=True), name="static")
