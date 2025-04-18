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
import streamlit.components.v1 as components
import base64

# Set up Streamlit page
st.set_page_config(page_title="Smart Number Plate Detection with Browser Webcam", layout="centered", initial_sidebar_state="expanded")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "model" not in st.session_state:
    st.session_state["model"] = None
if "reader" not in st.session_state:
    st.session_state["reader"] = None

USER_CREDENTIALS = {
    "admin": "admin123",
    "user1": "password123",
}

def encrypt_data(data):
    hashed_data = {}
    for plate, details in data.items():
        plate_hash = hashlib.sha256(plate.encode()).hexdigest()
        hashed_data[plate_hash] = details
    return hashed_data

encrypted_stolen_plates = encrypt_data({
    "TN01AB1234": "Reported stolen - Chennai",
    "KA09XY9876": "Police Alert - Bengaluru",
    "MH12ZZ0001": "Missing vehicle - Pune"
})

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

def is_malicious_image(file):
    file.seek(0)
    header_type = imghdr.what(None, h=file.read(512))
    file.seek(0)
    return header_type not in ['jpeg', 'png']

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

def draw_detections(frame, detections):
    for x1, y1, x2, y2, plate_text, is_stolen in detections:
        color = (0, 0, 255) if is_stolen else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def decode_base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    return img

# 📌 Corrected — Enlarged browser webcam component
def browser_webcam_component(conf_threshold):
    st.title("📷 Browser Webcam Detection")

    st.markdown("""
    <h5>Click "Capture Frame" and see detection results below:</h5>
    """, unsafe_allow_html=True)

    # Hidden text_input to receive Base64 frame from JS
    captured_frame = st.text_input("Captured Frame (Base64)", key="captured_frame")

    webcam_html = """
    <script>
      const video = document.createElement('video');
      video.style.width = '90%%';
      video.style.maxWidth = '800px';
      video.style.border = '3px solid black';
      video.style.borderRadius = '10px';
      video.autoplay = true;
      video.playsInline = true;
      document.body.appendChild(video);

      const button = document.createElement('button');
      button.innerText = 'Capture Frame';
      button.style.marginTop = '20px';
      button.style.padding = '14px 28px';
      button.style.fontSize = '18px';
      button.style.border = 'none';
      button.style.borderRadius = '5px';
      button.style.backgroundColor = '#3498db';
      button.style.color = 'white';
      button.style.cursor = 'pointer';
      button.onmouseover = () => button.style.backgroundColor = '#2980b9';
      button.onmouseout = () => button.style.backgroundColor = '#3498db';
      document.body.appendChild(button);

      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => { video.srcObject = stream; })
        .catch(error => { console.error("Error accessing webcam:", error); });

      button.addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg').split(',')[1];
        const input = window.parent.document.querySelector("input[data-baseweb='input']");
        if(input){
            input.value = frameData;
            input.dispatchEvent(new Event("input", { bubbles: true }));
        }
      });
    </script>
    """
    components.html(webcam_html, height=800)

    if captured_frame:
        st.success("✅ Frame captured successfully!")
        img = decode_base64_to_image(captured_frame)
        detections = detect_number_plate(img, conf_threshold)
        result_frame = draw_detections(img, detections)

        for _, _, _, _, plate_text, is_stolen in detections:
            if is_stolen:
                st.error(f"🚨 ALERT: {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")

        st.image(result_frame, channels="BGR", caption="Detected Frame")


def detection_system():
    st.title("🚘 Smart Number Plate Detection System")

    if st.session_state["model"] is None:
        st.session_state["model"] = YOLO("yolov8n.pt")
    if st.session_state["reader"] is None:
        st.session_state["reader"] = easyocr.Reader(['en'])

    st.sidebar.header("Choose Input Mode")
    input_type = st.sidebar.radio("Select input type", ["Image", "Video", "Browser Webcam", "Directory (ZIP)"])
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.5, 0.05)

    if input_type == "Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            if is_malicious_image(uploaded_image):
                st.error("❌ Uploaded file is not a valid image or may be malicious.")
            else:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                detections = detect_number_plate(frame, conf_threshold)
                result_frame = draw_detections(frame, detections)
                for _, _, _, _, plate_text, is_stolen in detections:
                    if is_stolen:
                        st.error(f"🚨 ALERT: {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")
                st.image(result_frame, channels="BGR", caption="Processed Image")

    elif input_type == "Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            progress_bar = st.progress(0)
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detect_number_plate(frame, conf_threshold)
                result_frame = draw_detections(frame, detections)
                for _, _, _, _, plate_text, is_stolen in detections:
                    if is_stolen:
                        st.warning(f"🚨 ALERT: {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")
                stframe.image(result_frame, channels="BGR")
                frame_count += 1
                progress_bar.progress((frame_count % 100) / 100)
            cap.release()
            progress_bar.empty()

    elif input_type == "Browser Webcam":
        browser_webcam_component(conf_threshold)

    elif input_type == "Directory (ZIP)":
        uploaded_zip = st.file_uploader("Upload a ZIP file of images", type=["zip"])
        if uploaded_zip:
            with tempfile.TemporaryDirectory() as extract_dir:
                with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                image_files = [os.path.join(root, file) for root, _, files in os.walk(extract_dir)
                               for file in files if file.lower().endswith(('png', 'jpg', 'jpeg'))]
                st.success(f"✅ Found {len(image_files)} image(s).")
                for img_path in image_files:
                    frame = cv2.imread(img_path)
                    detections = detect_number_plate(frame, conf_threshold)
                    result_frame = draw_detections(frame, detections)
                    for _, _, _, _, plate_text, is_stolen in detections:
                        if is_stolen:
                            st.error(f"🚨 ALERT: {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")
                    st.image(result_frame, channels="BGR", caption=os.path.basename(img_path))

if st.session_state["authenticated"]:
    detection_system()
else:
    login()
