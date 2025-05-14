import streamlit as st
import json
import torch
from PIL import Image, ImageDraw
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
from pymilvus import MilvusClient
import hashlib
from collections import Counter
from time import sleep

# Declare accepted filetypes
accepted_filetypes = (".jpeg", ".jpg", ".png", ".gif", ".tiff", ".tif", ".bmp", ".webp")

with open('known_frequencies.json', 'r') as f:
    known_frequencies = json.load(f)

class DatabaseConnectionError(Exception):
    pass

def initialize_database():
    try:
        client = MilvusClient("image-vectordb.db")
        collections = ["waterfall_embeddings", "fft_embeddings"]
        for collection in collections:
            if not client.has_collection(collection_name=collection):
                client.create_collection(
                    collection_name=collection,
                    dimension=512,
                    auto_id=True
                )
        return client
    except Exception as e:
        raise DatabaseConnectionError(f"Failed to initialize database: {e}")

# Initialize Milvus client
client = initialize_database()

# Check if the waterfall collection exists
if not client.has_collection(collection_name="waterfall_embeddings"):
    # Create the collection if it doesn't exist
    client.create_collection(
        collection_name="waterfall_embeddings",
        dimension=512,
        auto_id=True
    )

# Check if the fft collection exists
if not client.has_collection(collection_name="fft_embeddings"):
    # Create the collection if it doesn't exist
    client.create_collection(
        collection_name="fft_embeddings",
        dimension=512,
        auto_id=True
    )

# Feature Extractor class
class FeatureExtractor:
    def __init__(self, modelname):
        # First create model with num_classes matching training
        self.model = timm.create_model(modelname, 
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
        
        # Remove the classification head since we don't need it
        self.model.reset_classifier(num_classes=0, global_pool="avg")
        
        # Rest of your initialization code
        self.model.eval()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        print(f"Model is running on: {self.device}")
        
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

    def __call__(self, image):
        if isinstance(image, str):
            input_image = Image.open(image).convert("L").convert("RGB")
        else:
            input_image = image.convert("L").convert("RGB")
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)

        feature_vector = output.squeeze().cpu().numpy()
        normalized_vector = normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
        return normalized_vector

@st.cache_data
def load_model():
    return FeatureExtractor("resnet34")

extractor = load_model()

def search_images(query_image, collection):
    results = client.search(
        collection,
        data=[extractor(query_image)],
        output_fields=["filename", "class", "hash"],
        params={"metric_type": "COSINE"},
        limit=21
    )

    images = []
    for result in results:
        for hit in result:
            filename = hit["entity"]["filename"]
            try:
                img = Image.open(filename)
                img = img.resize((150, 150))
                similarity_score = hit["distance"]
                signal_class = hit["entity"]["class"]
                filehash = hit["entity"]["hash"]
                images.append((filename, similarity_score, signal_class, filehash, img))
            except FileNotFoundError:
                print(f"File not found: {filename}\nDid you delete the file from the datasets folder?")

    return images

def confidence_score(query_image, top_k_images, similarity_threshold):
    st.image(query_image.resize((150, 150)), caption="Query Image")

    # Limit to top 20 images
    top_k_images = top_k_images[:20]

    width = 150 * 5
    height = 150 * 4  # Increased to accommodate up to 20 images

    concatenated_image = Image.new("RGB", (width, height))
    filtered_images = []

    for idx, (filename, similarity_score, signal_class, filehash, img) in enumerate(top_k_images):
        if similarity_score >= similarity_threshold:
            filtered_images.append((filename, similarity_score, signal_class, filehash, img))
            x = idx % 5
            y = idx // 5
            concatenated_image.paste(img, (x * 150, y * 150))
            draw = ImageDraw.Draw(concatenated_image)
            draw.text((x * 150, y * 150 + 110), f"Class: {signal_class}", fill=(255, 255, 255))
            draw.text((x * 150, y * 150 + 125), f"Score: {similarity_score:.2f}", fill=(255, 255, 255))

    if filtered_images:
        result_classes = [img[2] for img in filtered_images]
        class_counts = Counter(result_classes)
        total_count = len(result_classes)
        confidence_scores = {cls: count / total_count * 100 for cls, count in class_counts.items()}

        st.write("### Confidence Scores and Frequency Ranges:")
        # Sort confidence scores in descending order
        sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        for cls, score in sorted_scores:
            freq_range = get_frequency_range(cls)
            with st.expander(f"{cls}: {score:.2f}%"):
                st.markdown(f"Frequency Range:  \n{freq_range}")
        st.image(concatenated_image, caption="Similar Signals")
    else:
        st.write("No signals meet the similarity threshold.")

def embed_images(folder_path, collection):
    batch_size = 500
    batch_data = []
    with st.status("Embedding and Inserting Documents. This may take a while.  Please do not stop the application.", expanded=True) as status:
        for dirpath, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith(accepted_filetypes):
                    filepath = os.path.join(dirpath, filename)
                    with open(filepath, "rb") as file:
                        file_hash = hashlib.sha256(file.read()).hexdigest()

                    results = client.search(
                        collection,
                        data=[extractor(filepath)],
                        output_fields=["filename", "hash"],
                        params={"metric_type": "COSINE"},
                    )
                    if results and results[0] and any(hit["entity"]["hash"] == file_hash for hit in results[0]):
                        continue

                    image_embedding = extractor(filepath)
                    batch_data.append({
                        "vector": image_embedding.tolist(),
                        "filename": filepath,
                        "class": os.path.basename(os.path.dirname(filepath)),
                        "hash": file_hash
                    })

                    if len(batch_data) >= batch_size:
                        try:
                            insert_result = client.insert(collection, batch_data)
                            st.write(f"Inserted {insert_result['insert_count']} entities")
                        except Exception as e:
                            st.write(f"Error inserting batch: {e}")
                        batch_data = []
        status.update(label="Finished Embedding Documents!", state="complete", expanded=False)

    if batch_data:
        try:
            insert_result = client.insert(collection, batch_data)
            st.write(f"Inserted {insert_result['insert_count']} entities")
        except Exception as e:
            st.write(f"Error inserting final batch: {e}")

def delete_image(identifier, identifier_type):
    try:
        if identifier_type == "hash":
            expr = f'hash == "{identifier}"'
        else:  # filename
            # Use 'like' operator for partial match
            expr = f'filename like "%/{identifier}"'
        
        delete_waterfall = client.delete(collection_name="waterfall_embeddings", filter=expr)
        delete_fft = client.delete(collection_name="fft_embeddings", filter=expr)
        
        if int(delete_waterfall[0]) >= 0 or int(delete_fft[0]) >= 0:
            st.success(f"Successfully deleted image(s).")
        else:
            st.warning("No matching images found to delete.")
    except Exception as e:
        st.error(f"Error deleting image: {e}")

def query_image(identifier, identifier_type):
    try:
        if identifier_type == "hash":
            expr = f'hash == "{identifier}"'
        else:  # filename
            # Use 'like' operator for partial match
            expr = f'filename like "%/{identifier}"'
        
        query_results = client.query(collection_name="waterfall_embeddings", filter=expr)
        if not query_results: query_results = client.query(collection_name="fft_embeddings", filter=expr)
        
        if query_results:
            st.success(f"Found {len(query_results)} matching image(s).")
            
            for result in query_results:
                filename = result['filename']
                signal_class = result['class']
                filehash = result['hash']
                
                # Open and resize the image
                img = Image.open(filename)
                img = img.resize((150, 150))
                
                # Display the image with its details
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(img, caption="Query Result")
                with col2:
                    st.write(f"**Class:** {signal_class}")
                    st.write(f"**Hash:** {filehash}")
                
                st.markdown("---")  # Add a separator between images
        else:
            st.warning("No matching images found.")
    except Exception as e:
        st.error(f"Error querying image: {e}")
    
@st.cache_data
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

    freq_range = known_frequencies.get(signal_class)
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

@st.cache_data
def identify_frequency(freq):
    # Check if freq is a valid number
    if not isinstance(freq, (int, float)) or freq < 0:
        return ["Invalid frequency"]
    
    def in_range(min, max):
        forgiveness = 0.03
        return freq >= min * (1-forgiveness) and freq <= max * (1+forgiveness)

    possible_signals = []
    for signal_type, freq_range in known_frequencies.items():
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


# Streamlit app
def main():
    st.title("Classidyne Signal Classifier")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Signal Classification", "Upload Signal Images", "Delete Image", "Image Search", "Signal Type Search"])


    with tab1:
        st.header("Signal Classification")
        option_map = {
        0: "Waterfall",
        1: "FFT",
        2: "Both",
        }
        upload_mode = st.segmented_control(
        "Search by",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        )
        if upload_mode != 1:
            uploaded_file = st.file_uploader("Choose a waterfall image...", type=list(accepted_filetypes))
            # Similarity threshold slider
            similarity_threshold = st.slider("Minimum Waterfall Similarity Score", 0.0, 1.0, 0.5, 0.01)
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                top_k = search_images(image, "waterfall_embeddings")
                confidence_score(image, top_k, similarity_threshold)

        if upload_mode != 0:
            uploaded_file = st.file_uploader("Choose a fft image...", type=list(accepted_filetypes))
            # Similarity threshold slider
            similarity_threshold = st.slider("Minimum FFT Similarity Score", 0.0, 1.0, 0.5, 0.01)
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                top_k = search_images(image, "fft_embeddings")
                confidence_score(image, top_k, similarity_threshold)


    with tab2:
        st.write("""
            ## Adding New RF Signal Images

            Follow these steps to add new images to the vector database:

            1. **Organize your images**: Place each image in the correct folder based on its signal type.
            - Example: 'datasets/waterfall/wifi/' for waterfall WiFi signals, 'datasets/fft/bluetooth/' for fft Bluetooth signals.

            2. **Folder structure**: Ensure all images are in folders following this pattern:
            'datasets/waterfall/signal-type/' and 'datasets/fft/signal-type'.
            Where 'signal-type' is the specific type of signal (e.g., wifi, bluetooth).

            3. **Unknown signals**: If you're unsure of the signal type, place the image in 'datasets/image-type/unknown/'.

            4. **Upload**: Once you've organized your images, click the 'Upload' button to add them to the signal database.
        
            5. **Creating New Signal Folders**:  You can create new signal folders by navigating to the 'datasets' folder and creating a new folder with the desired signal type.  For example, to create a new 'Zigbee' folder for Zigbee signals, navigate to 'datasets' and create a folder named 'zigbee'.
                 
            6. **To reset the database, delete the 'image-vectordb.db' file before starting the embedding process.**
            
            **Note**: All signal image folders must be located within the 'datasets' directory.  
                 
            *Supported image formats: jpg, jpeg, png, gif, tiff, tif, bmp, webp*
        """)
        waterfall_path = "./datasets/waterfall"
        fft_path = "./datasets/fft"

        if 'embedding_in_progress' not in st.session_state:
            st.session_state.embedding_in_progress = False

        def start_embedding():
            st.session_state.embedding_in_progress = True
            if waterfall_path:
                st.write("Starting waterfall embedding...")
                embed_images(waterfall_path, "waterfall_embeddings")
            else:
                st.write("Invalid waterfall image path. Consider insuring correct path. Skipping...")
            if fft_path:
                st.write("Starting FFT embedding...")
                embed_images(fft_path, "fft_embeddings")
            else:
                st.write("Invalid fft path. Consider insuring correct path. Skipping...")
            st.session_state.embedding_in_progress = False

        placeholder = st.empty()
        upload_button = placeholder.button("Upload", disabled=st.session_state.embedding_in_progress)

        if upload_button and not st.session_state.embedding_in_progress:
            placeholder.empty()
            st.info("Embedding in progress. Please wait...")
            start_embedding()
            st.info("Embedding complete...")
            sleep(10)
            st.rerun()
            
    
    with tab3:
        st.header("Delete Image")
        delete_option = st.radio("Delete by:", ("Hash", "Filename"))
        
        if delete_option == "Hash":
            identifier = st.text_input("Enter SHA256 hash:")
            identifier_type = "hash"
        else:
            identifier = st.text_input("Enter filename:")
            identifier_type = "filename"
        
        if st.button("Delete"):
            if identifier:
                delete_image(identifier, identifier_type)
            else:
                st.warning("Please enter a valid hash or filename.")


    with tab4:
        st.header("Image Search")
        search_option = st.radio("Search by:", ("Hash", "Filename"))
        if search_option == "Hash":
            identifier = st.text_input("Enter the SHA256 hash:")
            identifier_type = "hash"
        else:
            identifier = st.text_input("Enter the filename:")
            identifier_type = "filename"

        if st.button("Search"):
            if identifier:
                query_image(identifier, identifier_type)
            else:
                st.warning("Please enter a valid hash or filename.")


    with tab5:
        waterfall_path = "./datasets/waterfall"
        fft_path = "./datasets/fft"

        @st.cache_data
        def make_block(path, rf_type):
                path = os.path.join(path, rf_type)
                images = [Image.open(os.path.join(path, img) ).resize((150,150)) for img in os.listdir(os.path.join(path)) if img.lower().endswith(accepted_filetypes)][:25]
                concatenated_image = Image.new("RGB", (150 * 5, 150 * 5))
                for idx, img in enumerate(images):
                    x = idx % 5
                    y = idx // 5
                    concatenated_image.paste(img, (x * 150, y * 150))
                return concatenated_image

        st.header("Signal Type Search")
        type_viewer, frequency_search = st.tabs(["Type Viewer", "Frequency  Search"])
        rf_types = [d for d in os.listdir(waterfall_path) if os.path.isdir(os.path.join(waterfall_path, d))]

        # Type viewing function
        with type_viewer:
            rf_type = st.selectbox("View Images", rf_types)
            try:
                st.image(make_block(waterfall_path, rf_type), caption=f"Waterfalls from {rf_type} signal type.")
            except:
                pass
            try:
                st.image(make_block(fft_path, rf_type), caption=f"FFTs from {rf_type} signal type.")
            except:
                pass
        
        # Frequency search function
        with frequency_search:
           

            unit_conversions = {"Hz":1, "KHz":1e3, "MHz":1e6, "GHz":1e9}
            frequency = st.number_input("Enter the observed frequency:", min_value=0.000, value=100.000, format="%0.4f", step=1.0)

            conv = st.segmented_control("Unit:", options=unit_conversions.keys(), default="MHz", selection_mode="single")

            if conv:  # if conversion unit selected, apply it to the frequency
                frequency *= unit_conversions[conv]
                with st.spinner():
                    signal_types = identify_frequency(frequency)
                    if len(signal_types) != 0:
                        for signal in signal_types:
                            freq_range = get_frequency_range(signal)
                            with st.expander(signal):
                                st.write(freq_range)
                                try:
                                    waterfall_image = make_block(waterfall_path, signal)
                                    if waterfall_image:
                                        st.image(waterfall_image, caption=f"Waterfalls from {signal} signal type.")
                                    else:
                                        st.write(f"No waterfall images found for {signal} signal type.")
                                except Exception as e:
                                    st.write(f"Error loading waterfall images for {signal}: {str(e)}")
                                
                                try:
                                    fft_image = make_block(fft_path, signal)
                                    if fft_image:
                                        st.image(fft_image, caption=f"FFTs from {signal} signal type.")
                                    else:
                                        st.write(f"No FFT images found for {signal} signal type.")
                                except Exception as e:
                                    st.write(f"Could not loading FFT images for {signal}")
                    else:
                        st.write("No types found for that frequency")
            else:
                st.write("Please select a unit")


if __name__ == "__main__":
    main()
