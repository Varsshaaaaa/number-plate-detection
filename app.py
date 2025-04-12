import streamlit as st
import cv2
import numpy as np
import tempfile
import easyocr
from ultralytics import YOLO
from streamlit_lottie import st_lottie
import requests
import zipfile
import os
from PIL import Image

# Streamlit page config
st.set_page_config(page_title="Smart Number Plate Detection", layout="centered")

# UI Enhancements
st.markdown("""
    <style>
    .title h1 {
        font-size: 2.5em;
        background: linear-gradient(90deg, #1db954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in-out;
        text-align: center;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .stFileUploader > label {
        border: 2px dashed #1ed760;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        background-color: #f9fff9;
        transition: all 0.3s ease-in-out;
    }
    .stFileUploader > label:hover {
        background-color: #eaffea;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"><h1>🚘 Smart Number Plate Detection System</h1></div>', unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with st.sidebar:
    lottie_car = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
    st_lottie(lottie_car, height=180, key="car-anime")

# Load model and OCR reader
model = YOLO("yolov8n.pt")  # You can replace this with a custom-trained number plate model
reader = easyocr.Reader(['en'])

# Sample stolen plate database
stolen_plates = {
    "TN01AB1234": "Reported stolen - Chennai",
    "KA09XY9876": "Police Alert - Bengaluru",
    "MH12ZZ0001": "Missing vehicle - Pune"
}

# Sidebar options
st.sidebar.header("Choose Input Mode")
input_type = st.sidebar.radio("Select input type", ["Image", "Video", "Webcam", "Directory (ZIP)"])
conf_threshold = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.5, 0.05)

# Plate Detection Function
def detect_number_plate(frame):
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        if box.conf[0] >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            text = reader.readtext(plate_img, detail=0)
            plate_text = ''.join(text).replace(' ', '').upper()
            is_stolen = plate_text in stolen_plates
            detections.append((x1, y1, x2, y2, plate_text, is_stolen))
    return detections

# Draw detections
def draw_detections(frame, detections):
    for x1, y1, x2, y2, plate_text, is_stolen in detections:
        color = (0, 0, 255) if is_stolen else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# Image Input
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        detections = detect_number_plate(frame)
        result_frame = draw_detections(frame, detections)
        for _, _, _, _, plate_text, is_stolen in detections:
            if is_stolen:
                st.error(f"🚨 ALERT: {plate_text} - {stolen_plates[plate_text]}")
        st.image(result_frame, channels="BGR", caption="Processed Image")

# Video Input
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = detect_number_plate(frame)
            result_frame = draw_detections(frame, detections)
            for _, _, _, _, plate_text, is_stolen in detections:
                if is_stolen:
                    st.error(f"🚨 ALERT: {plate_text} - {stolen_plates[plate_text]}")
            stframe.image(result_frame, channels="BGR")
        cap.release()

# Webcam Input
elif input_type == "Webcam":
    st.warning("Please run this locally to use webcam feature.")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detect_number_plate(frame)
            result_frame = draw_detections(frame, detections)
            for _, _, _, _, plate_text, is_stolen in detections:
                if is_stolen:
                    st.error(f"🚨 ALERT: {plate_text} - {stolen_plates[plate_text]}")
            stframe.image(result_frame, channels="BGR")
        cap.release()

# Directory of Images from ZIP
elif input_type == "Directory (ZIP)":
    uploaded_zip = st.file_uploader("Upload a ZIP file of images", type=["zip"])
    if uploaded_zip is not None:
        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            image_files = []
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))

            st.success(f"✅ {len(image_files)} image(s) found in the ZIP (including subdirectories)")

            for img_path in image_files:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                detections = detect_number_plate(frame)
                result_frame = draw_detections(frame, detections)

                for _, _, _, _, plate_text, is_stolen in detections:
                    if is_stolen:
                        st.error(f"🚨 ALERT: {plate_text} - {stolen_plates[plate_text]}")

                st.image(result_frame, channels="BGR", caption=os.path.basename(img_path))
