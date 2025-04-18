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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Streamlit page config
st.set_page_config(page_title="Smart Number Plate Detection", layout="wide")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "model" not in st.session_state:
    st.session_state["model"] = None
if "reader" not in st.session_state:
    st.session_state["reader"] = None

# Predefined credentials
USER_CREDENTIALS = {
    "admin": "admin123",
    "user1": "password123",
}

# Encrypt stolen plates
def encrypt_data(data):
    hashed_data = {}
    for plate, details in data.items():
        plate_hash = hashlib.sha256(plate.encode()).hexdigest()
        hashed_data[plate_hash] = details
    return hashed_data

# Stolen vehicles DB
encrypted_stolen_plates = encrypt_data({
    "TN01AB1234": "Reported stolen - Chennai",
    "KA09XY9876": "Police Alert - Bengaluru",
    "MH12ZZ0001": "Missing vehicle - Pune"
})

# Basic login
def login():
    st.title("🔒 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials.")

# Detect number plates in frame
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

# Draw bounding boxes
def draw_detections(frame, detections):
    for x1, y1, x2, y2, plate_text, is_stolen in detections:
        color = (0, 0, 255) if is_stolen else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

# Video Processor for webcam
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        detections = detect_number_plate(img, self.conf_threshold)
        img = draw_detections(img, detections)
        return img

# Main app
def detection_system():
    st.title("🚘 Smart Number Plate Detection")

    if st.session_state["model"] is None:
        st.session_state["model"] = YOLO("yolov8n.pt")
    if st.session_state["reader"] is None:
        st.session_state["reader"] = easyocr.Reader(['en'])

    st.sidebar.header("Choose Input")
    input_type = st.sidebar.radio("Input type", ["Image", "Video", "Webcam", "Directory (ZIP)"])
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.5, 0.05)

    if input_type == "Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            if imghdr.what(uploaded_image) not in ['jpeg', 'png']:
                st.error("Invalid image.")
            else:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                detections = detect_number_plate(frame, conf_threshold)
                result_frame = draw_detections(frame, detections)
                for _, _, _, _, plate_text, is_stolen in detections:
                    if is_stolen:
                        st.error(f"🚨 {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")
                st.image(result_frame, channels="BGR")

    elif input_type == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detect_number_plate(frame, conf_threshold)
                result_frame = draw_detections(frame, detections)
                stframe.image(result_frame, channels="BGR")
            cap.release()

    elif input_type == "Webcam":
        st.write("Live Webcam Stream")
        webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        )

    elif input_type == "Directory (ZIP)":
        uploaded_zip = st.file_uploader("Upload ZIP of images", type=["zip"])
        if uploaded_zip:
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                image_files = [os.path.join(root, file)
                               for root, _, files in os.walk(extract_dir)
                               for file in files if file.lower().endswith(('png', 'jpg', 'jpeg'))]
                st.success(f"Found {len(image_files)} images.")
                for img_path in image_files:
                    frame = cv2.imread(img_path)
                    detections = detect_number_plate(frame, conf_threshold)
                    result_frame = draw_detections(frame, detections)
                    for _, _, _, _, plate_text, is_stolen in detections:
                        if is_stolen:
                            st.error(f"🚨 {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")
                    st.image(result_frame, channels="BGR", caption=os.path.basename(img_path))

# Run detection system if logged in
if st.session_state["authenticated"]:
    detection_system()
else:
    login()
