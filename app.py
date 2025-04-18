import streamlit as st
import cv2
import numpy as np
import tempfile
import easyocr
from ultralytics import YOLO
import hashlib
import os
import zipfile
import imghdr
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set up the Streamlit page
st.set_page_config(page_title="Smart Number Plate Detection with Login", layout="centered", initial_sidebar_state="expanded")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "model" not in st.session_state:
    st.session_state["model"] = None
if "reader" not in st.session_state:
    st.session_state["reader"] = None

# Predefined user credentials
USER_CREDENTIALS = {
    "admin": "admin123",
    "user1": "password123",
}

# Encrypt license plates
def encrypt_data(data):
    hashed_data = {}
    for plate, details in data.items():
        plate_hash = hashlib.sha256(plate.encode()).hexdigest()
        hashed_data[plate_hash] = details
    return hashed_data

# Stolen vehicle plate data
encrypted_stolen_plates = encrypt_data({
    "TN01AB1234": "Reported stolen - Chennai",
    "KA09XY9876": "Police Alert - Bengaluru",
    "MH12ZZ0001": "Missing vehicle - Pune"
})

# Basic login UI
def login():
    st.title("🔒 Login to Access Detection System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password. Please try again.")

# Check if uploaded file is a valid image
def is_malicious_image(file):
    file.seek(0)
    header_type = imghdr.what(None, h=file.read(512))
    file.seek(0)
    return header_type not in ['jpeg', 'png']

# Detect number plates in frames
def detect_number_plate(frame, conf_threshold):
    model = st.session_state["model"]
    reader = st.session_state["reader"]
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        if box.conf[0] >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            text = reader.readtext(plate_img, detail=0)
            plate_text = ''.join(text).replace(' ', '').upper()
            plate_hash = hashlib.sha256(plate_text.encode()).hexdigest()
            is_stolen = plate_hash in encrypted_stolen_plates
            detections.append((x1, y1, x2, y2, plate_text, is_stolen))
    return detections

# Draw bounding boxes with plate info
def draw_detections(frame, detections):
    for x1, y1, x2, y2, plate_text, is_stolen in detections:
        color = (0, 0, 255) if is_stolen else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Main detection UI logic
def detection_system():
    st.title("🚘 Smart Number Plate Detection System")

    # Ensure model and reader are initialized
    if st.session_state["model"] is None:
        st.session_state["model"] = YOLO("yolov8n.pt")  # Default YOLO model
    if st.session_state["reader"] is None:
        st.session_state["reader"] = easyocr.Reader(['en'])

    st.sidebar.header("Choose Input Mode")
    input_type = st.sidebar.radio("Select input type", ["Image", "Video", "Webcam", "Directory (ZIP)"])
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.5, 0.05)

    if input_type == "Webcam":
        st.subheader("📷 Real-time Detection via Webcam")

        class PlateDetectionTransformer(VideoTransformerBase):
            def __init__(self):
                # Ensure keys are initialized to prevent KeyError
                if "model" not in st.session_state or st.session_state["model"] is None:
                    st.session_state["model"] = YOLO("yolov8n.pt")
                if "reader" not in st.session_state or st.session_state["reader"] is None:
                    st.session_state["reader"] = easyocr.Reader(['en'])
                self.model = st.session_state["model"]
                self.reader = st.session_state["reader"]

            def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                detections = detect_number_plate(img, conf_threshold)
                result_img = draw_detections(img, detections)
                return av.VideoFrame.from_ndarray(result_img, format="bgr24")

        webrtc_streamer(
            key="number-plate",
            video_transformer_factory=PlateDetectionTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )

    # Add remaining logic for other input modes (Image, Video, ZIP)

# Entry point
if st.session_state["authenticated"]:
    detection_system()
else:
    login()
