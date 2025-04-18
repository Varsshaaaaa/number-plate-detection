import streamlit as st
import cv2
import numpy as np
import easyocr
import hashlib
from ultralytics import YOLO

# Set up the Streamlit page
st.set_page_config(page_title="Smart Number Plate Detection", layout="centered", initial_sidebar_state="expanded")

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
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

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

    if st.session_state["model"] is None:
        st.session_state["model"] = YOLO("yolov8n.pt")  # You can change this to a custom model
    if st.session_state["reader"] is None:
        st.session_state["reader"] = easyocr.Reader(['en'])

    # OpenCV Webcam Capture
    run_webcam = st.button("Start Webcam")
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame. Please check your webcam.")
                break

            conf_threshold = 0.5  # Set your confidence threshold
            detections = detect_number_plate(frame, conf_threshold)
            result_frame = draw_detections(frame, detections)

            # Show the processed frame in Streamlit
            stframe.image(result_frame, channels="BGR", use_column_width=True)

            # Check for stolen plates
            for _, _, _, _, plate_text, is_stolen in detections:
                if is_stolen:
                    st.warning(f"🚨 ALERT: {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")

        cap.release()

# Entry point
if st.session_state["authenticated"]:
    detection_system()
else:
    login()
