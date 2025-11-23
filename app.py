import streamlit as st
import numpy as np
import tensorflow.lite as tflite 
from PIL import Image, UnidentifiedImageError
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Satellite Imagery Segmentation", page_icon="🌍", layout="wide")

# --- UI STYLING ---
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .main { background: #0e1117; color: white; }
    h1 { color: #4F8BF9; }
</style>
""", unsafe_allow_html=True)

# --- LOAD LITE MODEL (FAST ENGINE) ---
@st.cache_resource
def load_lite_model():
    # Load the optimized TFLite model
    try:
        interpreter = tflite.Interpreter(model_path="satellite_optimized.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model not found or corrupt. Make sure 'satellite_optimized.tflite' is in the folder.")
        st.stop()

# --- PREPROCESSING PIPELINE ---
def process_image(image):
    image = image.convert('RGB')
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (512, 512))
    img_norm = img_resized / 255.0
    # TFLite requires FLOAT32 specifically
    img_batch = np.expand_dims(img_norm, axis=0).astype(np.float32)
    return img_batch, img_resized

# --- FAST INFERENCE FUNCTION ---
def run_lite_inference(interpreter, input_batch):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_batch)

    # Run the calculation (Invoke)
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# --- DECODER ---
def decode_prediction(pred_mask):
    class_ids = np.argmax(pred_mask[0], axis=-1)
    colors = np.array([
        [0, 255, 255],   # Urban
        [255, 255, 0],   # Agri
        [255, 0, 255],   # Range
        [0, 255, 0],     # Forest
        [0, 0, 255],     # Water
        [255, 255, 255], # Barren
        [0, 0, 0]        # Unknown
    ])
    return colors[class_ids].astype('uint8')

# --- MAIN APP ---
st.title("🌍 Satellite Imagery Semantic Segmentation")
st.markdown("### Deep Learning Analysis Tool (TF-Lite Accelerated)")

interpreter = load_lite_model()

st.sidebar.header("Input Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        with st.spinner('Running Optimized Inference...'):
            # 1. Prepare Data
            input_batch, original_img = process_image(image)
            
            # 2. Run Lite Inference
            prediction = run_lite_inference(interpreter, input_batch)
            
            # 3. Decode
            rgb_mask = decode_prediction(prediction)

        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Imagery")
            st.image(original_img, use_column_width=True)
        with col2:
            st.subheader("Segmentation Mask")
            st.image(rgb_mask, use_column_width=True)
            
        # Analytics
        st.markdown("---")
        st.subheader("📊 Land Cover Analysis")
        class_names = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
        class_ids = np.argmax(prediction[0], axis=-1)
        total_pixels = class_ids.size
        cols = st.columns(4)
        for i in range(len(class_names)):
            count = np.sum(class_ids == i)
            if count > 0:
                pct = (count / total_pixels) * 100
                if pct > 1.0: 
                    cols[i % 4].metric(label=class_names[i], value=f"{pct:.1f}%")

    except UnidentifiedImageError:
        st.error("Corrupt image file.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("System Ready. Upload an image to begin.")