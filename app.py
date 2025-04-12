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
import hashlib
import hmac
import secrets
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file (if any)
load_dotenv()

# --- Security: Authentication ---
# Define credentials (in a real app, use environment variables)
USERS = {
    "admin": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin
    "police": "d7f7a811463157aeba43c2c48544d2cbcab35e22109b9e1cfb905a8517c396ba"  # police123
}

def verify_password(username, password):
    if username not in USERS:
        return False
    stored_hash = USERS[username]
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(stored_hash, password_hash)

# Streamlit page config
st.set_page_config(page_title="Smart Number Plate Detection", layout="centered")

# --- UI Enhancements ---
# Color Palette
primary_color = "#9b87f5"
secondary_color = "#7E69AB"
accent_color = "#1EAEDB"
background_color = "#F1F0FB"
alert_color = "#ea384c"

# CSS Styles
st.markdown(f"""
    <style>
    :root {{
        --primary-color: {primary_color};
        --secondary-color: {secondary_color};
        --accent-color: {accent_color};
        --background-color: {background_color};
        --alert-color: {alert_color};
    }}

    .main {{
        background-color: var(--background-color);
        padding: 20px;
        border-radius: 10px;
    }}

    .title h1 {{
        font-size: 2.5em;
        color: var(--primary-color);
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in-out;
    }}

    .card {{
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }}

    .alert-box {{
        background-color: var(--alert-color);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0% {{opacity: 1;}}
        50% {{opacity: 0.8;}}
        100% {{opacity: 1;}}
    }}

    @keyframes fadeIn {{
        from {{opacity: 0; transform: translateY(-10px);}}
        to {{opacity: 1; transform: translateY(0);}}
    }}

    .stFileUploader > label {{
        border: 2px dashed var(--primary-color);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        background-color: #f9fff9;
        transition: all 0.3s ease-in-out;
    }}
    .stFileUploader > label:hover {{
        background-color: #eaffea;
        transform: scale(1.02);
    }}
    </style>
""", unsafe_allow_html=True)

# Authentication flow
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown('<div class="title"><h1>🚘 Smart Number Plate Detection System</h1></div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if verify_password(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()  # Stop execution if not authenticated

# Main App (only shown when authenticated)
st.markdown('<div class="title"><h1>🚘 Smart Number Plate Detection System</h1></div>', unsafe_allow_html=True)

# Show logged in user
st.sidebar.success(f"Logged in as {st.session_state['username']}")
if st.sidebar.button("Logout"):
    st.session_state["authenticated"] = False
    st.experimental_rerun()

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

# --- Security: Input Validation ---
def validate_file(uploaded_file, file_type="image"):
    if uploaded_file is None:
        return False, "No file uploaded"

    # Size validation
    file_size = uploaded_file.size
    if file_type == "zip":
        if file_size > 200 * 1024 * 1024:  # 200 MB limit for ZIP files
            return False, "File size exceeds 200 MB limit"
        else:
            if file_size > 10 * 1024 * 1024:  # 10 MB limit for other files
                 return False, "File size exceeds 10 MB limit"


    # Type validation
    if file_type == "image":
        valid_types = ["image/jpeg", "image/png", "image/jpg"]
        if uploaded_file.type not in valid_types:
            return False, "Invalid file type. Please upload JPEG or PNG images only."
    elif file_type == "video":
        valid_types = ["video/mp4", "video/mov", "video/avi"]
        if uploaded_file.type not in valid_types:
            return False, "Invalid file type. Please upload MP4, MOV or AVI videos only."
    elif file_type == "zip":
        if uploaded_file.type != "application/zip":
            return False, "Invalid file type. Please upload ZIP files only."

    return True, "File validation successful"

# --- Secure File Handling ---
def secure_temp_file(uploaded_file):
    # Generate secure random filename
    random_suffix = secrets.token_hex(8)
    file_extension = os.path.splitext(uploaded_file.name)[1]
    secure_filename = f"secure_temp_{random_suffix}{file_extension}"

    # Create secure temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, secure_filename)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Return path and cleanup function
    def cleanup():
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return temp_path, cleanup

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
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            is_valid, message = validate_file(uploaded_image, "image")
            if not is_valid:
                st.error(message)
            else:
                with st.spinner("Processing image..."):
                    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, 1)
                    detections = detect_number_plate(frame)
                    result_frame = draw_detections(frame, detections)
                
                if detections:
                    tab1, tab2 = st.tabs(["Visualization", "Detailed Results"])
                    with tab1:
                        st.image(result_frame, channels="BGR", caption="Processed Image")
                    with tab2:
                        for i, (_, _, _, _, plate_text, is_stolen) in enumerate(detections):
                            st.markdown(f"*Plate #{i+1}:* {plate_text}")
                            if is_stolen:
                                st.markdown(f"<div class='alert-box'>🚨 ALERT: {stolen_plates[plate_text]}</div>", unsafe_allow_html=True)
                            else:
                                st.success("✓ No alerts for this plate")
                else:
                    st.image(result_frame, channels="BGR", caption="Processed Image")
                    st.info("No license plates detected in this image")
        st.markdown('</div>', unsafe_allow_html=True)

# Video Input
elif input_type == "Video":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            is_valid, message = validate_file(uploaded_video, "video")
            if not is_valid:
                st.error(message)
            else:
                with st.spinner("Processing video..."):
                    temp_path, cleanup = secure_temp_file(uploaded_video)
                    try:
                        cap = cv2.VideoCapture(temp_path)
                        stframe = st.empty()
                        alert_container = st.container()
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            detections = detect_number_plate(frame)
                            result_frame = draw_detections(frame, detections)
                            for _, _, _, _, plate_text, is_stolen in detections:
                                if is_stolen:
                                    with alert_container:
                                        st.markdown(f"<div class='alert-box'>🚨 ALERT: {plate_text} - {stolen_plates[plate_text]}</div>", unsafe_allow_html=True)
                            stframe.image(result_frame, channels="BGR")
                        cap.release()
                    finally:
                        cleanup()
        st.markdown('</div>', unsafe_allow_html=True)

# Webcam Input
elif input_type == "Webcam":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.warning("Please run this locally to use webcam feature.")
        if st.button("Start Webcam"):
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            alert_container = st.container()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detect_number_plate(frame)
                result_frame = draw_detections(frame, detections)
                for _, _, _, _, plate_text, is_stolen in detections:
                    if is_stolen:
                        with alert_container:
                            st.markdown(f"<div class='alert-box'>🚨 ALERT: {plate_text} - {stolen_plates[plate_text]}</div>", unsafe_allow_html=True)
                stframe.image(result_frame, channels="BGR")
            cap.release()
        st.markdown('</div>', unsafe_allow_html=True)

# Directory of Images from ZIP
elif input_type == "Directory (ZIP)":
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_zip = st.file_uploader("Upload a ZIP file of images", type=["zip"])
        if uploaded_zip is not None:
            is_valid, message = validate_file(uploaded_zip, "zip")
            if not is_valid:
                st.error(message)
            else:
                with st.spinner("Extracting and processing ZIP file..."):
                    with tempfile.TemporaryDirectory() as extract_dir:
                        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                            zip_ref.extractall(extract_dir)

                        image_files = []
                        for root, _, files in os.walk(extract_dir):
                            for file in files:
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    image_files.append(os.path.join(root, file))

                        st.success(f"✅ {len(image_files)} image(s) found in the ZIP (including subdirectories)")

                        alerts_found = False
                        for img_path in image_files:
                            frame = cv2.imread(img_path)
                            if frame is None:
                                continue
                            detections = detect_number_plate(frame)
                            result_frame = draw_detections(frame, detections)

                            for _, _, _, _, plate_text, is_stolen in detections:
                                if is_stolen:
                                    alerts_found = True
                                    st.markdown(f"<div class='alert-box'>🚨 ALERT in {os.path.basename(img_path)}: {plate_text} - {stolen_plates[plate_text]}</div>", unsafe_allow_html=True)

                            st.image(result_frame, channels="BGR", caption=os.path.basename(img_path))
                        
                        if not alerts_found:
                            st.success("✓ No alerts found in any of the images")
        st.markdown('</div>', unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #888;">
    <p>Developed with ❤ by Police Tech Team • Contact support: police@example.com</p>
    <p style="font-size: 0.8em;">© 2025 Police Department. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
