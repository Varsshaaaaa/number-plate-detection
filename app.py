import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import hashlib
import easyocr
from datetime import datetime
from cryptography.fernet import Fernet

# Load YOLO model and EasyOCR reader
model = YOLO("best.pt")
reader = easyocr.Reader(['en'])

# Stolen vehicle database and encryption
stolen_vehicles = {
    "TN01AB1234": "Vehicle reported stolen in Chennai",
    "KL05CD5678": "Vehicle reported stolen in Kochi"
}
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# Encrypt stolen plates for security
encrypted_stolen_plates = {hashlib.sha256(plate.encode()).hexdigest(): info for plate, info in stolen_vehicles.items()}

# Function to perform detection
def detect_number_plate(image, confidence_threshold):
    results = model(image)[0]
    detections = []
    for result in results.boxes:
        if result.conf >= confidence_threshold:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            plate_crop = image[y1:y2, x1:x2]
            plate_text = reader.readtext(plate_crop)
            plate_number = plate_text[0][1] if plate_text else "Unknown"
            encrypted_plate = hashlib.sha256(plate_number.encode()).hexdigest()
            is_stolen = encrypted_plate in encrypted_stolen_plates
            detections.append((x1, y1, x2, y2, plate_number, is_stolen))
    return detections

# Streamlit UI
st.title("🚔 Smart Number Plate Detection")
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.01)

mode = st.radio("Choose Input Mode", ("Image", "Webcam"))

if mode == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        detections = detect_number_plate(image_np, confidence_threshold)
        for x1, y1, x2, y2, plate_number, is_stolen in detections:
            color = (0, 0, 255) if is_stolen else (0, 255, 0)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if is_stolen:
                st.error(f"🚨 Stolen Vehicle Detected: {plate_number}")
        st.image(image_np, channels="RGB")

elif mode == "Webcam":
    picture = st.camera_input("Take a photo")

    if picture:
        image = Image.open(picture).convert("RGB")
        image_np = np.array(image)
        detections = detect_number_plate(image_np, confidence_threshold)
        for x1, y1, x2, y2, plate_number, is_stolen in detections:
            color = (0, 0, 255) if is_stolen else (0, 255, 0)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_np, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            if is_stolen:
                st.error(f"🚨 Stolen Vehicle Detected: {plate_number}")
        st.image(image_np, channels="RGB")
