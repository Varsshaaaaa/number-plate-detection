import streamlit as st
import numpy as np
import cv2
import easyocr
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import hashlib

# Initialize session state
if "model" not in st.session_state:
    st.session_state["model"] = YOLO("yolov8n.pt")  # Load YOLO model
if "reader" not in st.session_state:
    st.session_state["reader"] = easyocr.Reader(['en'])  # Initialize EasyOCR

# Function to process the captured image
def detect_number_plate(frame, conf_threshold=0.5):
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
            detections.append((x1, y1, x2, y2, plate_text))
    return detections

# Function to draw bounding boxes on detected plates
def draw_detections(frame, detections):
    for x1, y1, x2, y2, plate_text in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

# Class for handling webcam capture and frame processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = 0.5

    def transform(self, frame):
        # Convert the frame to an OpenCV-compatible format
        img = frame.to_ndarray(format="bgr24")

        # Process the frame to detect number plates
        detections = detect_number_plate(img, self.conf_threshold)
        result_frame = draw_detections(img, detections)

        return result_frame

# Streamlit App Layout
st.title("🚘 Smart Number Plate Detection System")

# Sidebar for settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.5, 0.05)

# Capture the video stream from the webcam
st.sidebar.header("Webcam Stream")
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Display information about detected number plates
st.markdown("""
    This system will capture video from your webcam and attempt to detect vehicle number plates in real-time. 
    If a number plate is detected, it will be highlighted with a bounding box and displayed with the plate number.
""")
