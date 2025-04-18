import streamlit as st
import cv2
import numpy as np
import easyocr
import hashlib
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Initialize models only when the user is authenticated
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Predefined user credentials (example)
USER_CREDENTIALS = {
    "admin": "admin123",
}

# Basic login function
def login():
    st.title("🔒 Login to Access Detection System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if USER_CREDENTIALS.get(username) == password:
            st.session_state["authenticated"] = True
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

# Function to detect plates in video frames
def detect_number_plate(frame, model, reader, conf_threshold=0.5):
    results = model(frame)[0]  # YOLO model inference
    detections = []
    for box in results.boxes:
        if box.conf[0] >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            text = reader.readtext(plate_img, detail=0)
            plate_text = ''.join(text).replace(' ', '').upper()
            detections.append((x1, y1, x2, y2, plate_text))
    return detections

# Function to draw bounding boxes and plate texts
def draw_detections(frame, detections):
    for x1, y1, x2, y2, plate_text in detections:
        color = (0, 255, 0)  # Green color for detected plates
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Video transformer class for webcam processing
class PlateDetectionTransformer(VideoTransformerBase):
    def __init__(self, model, reader):
        self.model = model
        self.reader = reader

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        detections = detect_number_plate(frame, self.model, self.reader)
        frame = draw_detections(frame, detections)
        return frame

# Main detection system
def detection_system():
    st.title("🚘 Smart Number Plate Detection System")
    model = YOLO("yolov8n.pt")  # Load YOLO model
    reader = easyocr.Reader(['en'])  # Load OCR reader

    input_type = st.selectbox("Select input type", ["Webcam", "Upload Image"])
    
    if input_type == "Webcam":
        st.subheader("📷 Real-time Detection via Webcam")
        webrtc_streamer(
            key="number-plate-webcam",
            video_transformer_factory=lambda: PlateDetectionTransformer(model, reader),
            media_stream_constraints={"video": True, "audio": False}
        )

    elif input_type == "Upload Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            detections = detect_number_plate(img, model, reader)
            result_img = draw_detections(img, detections)
            st.image(result_img, channels="BGR", caption="Processed Image")

# Entry point
if st.session_state["authenticated"]:
    detection_system()
else:
    login()
