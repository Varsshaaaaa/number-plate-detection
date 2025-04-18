import streamlit as st
import base64
import io
import numpy as np
import cv2
from PIL import Image
import easyocr
from ultralytics import YOLO

# Initialize Streamlit session state
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

# JavaScript + HTML for webcam capture
def capture_webcam():
    st.title("Webcam Access for Number Plate Detection")

    # HTML and JavaScript for webcam integration
    webcam_html = """
    <html>
    <head>
        <style>
            video {
                width: 100%;
                height: auto;
            }
            #captureButton {
                font-size: 20px;
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <h3>Webcam Stream</h3>
        <video id="webcam" autoplay></video>
        <br>
        <button id="captureButton">Capture Frame</button>

        <script>
            const webcam = document.getElementById('webcam');
            const captureButton = document.getElementById('captureButton');
            const canvas = document.createElement('canvas');
            canvas.style.display = 'none';
            document.body.appendChild(canvas);

            // Access webcam
            navigator.mediaDevices.getUserMedia({ video: true })
                .then((stream) => {
                    webcam.srcObject = stream;
                })
                .catch((error) => {
                    console.error("Error accessing webcam: ", error);
                });

            // Capture frame and send data to Streamlit
            captureButton.addEventListener('click', () => {
                canvas.width = webcam.videoWidth;
                canvas.height = webcam.videoHeight;
                const context = canvas.getContext('2d');
                context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg').split(',')[1]; // Base64 image data
                window.parent.postMessage({ type: 'image', data: imageData }, '*');
            });
        </script>
    </body>
    </html>
    """
    # Display the HTML content in Streamlit
    st.components.v1.html(webcam_html, height=500)

    # Get image data from JavaScript
    image_data = st.experimental_get_query_params().get('image')
    if image_data:
        # Decode the Base64 image data
        img_data = base64.b64decode(image_data[0])
        img = Image.open(io.BytesIO(img_data))
        frame = np.array(img)

        # Detect number plates
        detections = detect_number_plate(frame)
        result_frame = draw_detections(frame, detections)

        # Display processed image
        st.image(result_frame, channels="BGR", caption="Processed Image with Number Plate Detection")

# Main detection UI logic
def detection_system():
    st.title("🚘 Smart Number Plate Detection System")

    input_type = st.sidebar.radio("Select Input Type", ["Webcam", "Upload Image"])
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.25, 1.0, 0.5, 0.05)

    if input_type == "Webcam":
        capture_webcam()
    elif input_type == "Upload Image":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            detections = detect_number_plate(frame, conf_threshold)
            result_frame = draw_detections(frame, detections)
            st.image(result_frame, channels="BGR", caption="Processed Image")

# Run the detection system
detection_system()
