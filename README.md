# 🌍 Satellite Imagery Semantic Segmentation

A Deep Learning-based remote sensing tool that automates land-use classification from high-resolution satellite telemetry. This project demonstrates an end-to-end pipeline from **U-Net** training to **TensorFlow Lite** optimization and **Streamlit** deployment.

## 🚀 Project Overview

Manual analysis of satellite imagery is slow and prone to error. This system solves the **Data Latency** problem by autonomously segmenting Earth Observation (EO) data into actionable classes:
* **Water Bodies** (Resource monitoring)
* **Urban Areas** (City planning/encroachment)
* **Vegetation/Forest** (Deforestation tracking)
* **Agriculture** (Crop yield estimation)

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow, Keras, U-Net Architecture (Custom Encoder-Decoder)
* **Optimization:** TensorFlow Lite (Quantization for CPU inference)
* **Computer Vision:** OpenCV, Pillow
* **Deployment:** Streamlit (Web Interface)
* **Data Handling:** NumPy (Tensor manipulation)

## 📊 Features

* **Semantic Segmentation:** Pixel-level classification of 7 distinct terrain types.
* **Real-Time Inference:** Optimized TFLite engine reduces latency to <2 seconds on standard CPUs.
* **Quantitative Reporting:** Automatically calculates land cover percentages (e.g., *"Water: 12.5%, Urban: 30%"*) for mission planning.
* **Robust Pipeline:** Handles transparency channel errors and corrupt inputs gracefully.

## 📸 Screenshots

*(Add your screenshots here. Example: Raw Input vs. AI Prediction)*

## 🔧 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/satellite-imagery-segmentation.git](https://github.com/your-username/satellite-imagery-segmentation.git)
    cd satellite-imagery-segmentation
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**
    ```bash
    streamlit run app.py
    ```

4.  **Upload Imagery**
    * Upload any standard `.jpg` or `.png` satellite image.
    * The system will auto-convert and process the feed.

## 🧠 Model Architecture

* **Backbone:** U-Net (Convolutional Neural Network)
* **Input:** 512x512x3 (RGB Telemetry)
* **Output:** 512x512x7 (Softmax Probability Map)
* **Metric:** Dice Coefficient (Intersection over Union)

## 📜 License

This project is open-source and available under the MIT License.
