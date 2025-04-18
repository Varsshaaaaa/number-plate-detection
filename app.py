import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import base64
import hashlib
from io import BytesIO
from PIL import Image

# Dummy stolen plate data (hash: location)
encrypted_stolen_plates = {
    hashlib.sha256("TN01AB1234".encode()).hexdigest(): "Coimbatore",
    hashlib.sha256("KL07CD5678".encode()).hexdigest(): "Chennai"
}

# Dummy detection function — replace with your YOLO detection logic
def detect_number_plate(frame, conf_threshold=0.5):
    # This is just a placeholder for demo purposes
    h, w, _ = frame.shape
    return [(int(w/4), int(h/4), int(w/2), int(h/2), "TN01AB1234", True)]  # (x1, y1, x2, y2, plate_text, is_stolen)

# Draw detections
def draw_detections(frame, detections):
    for (x1, y1, x2, y2, plate_text, is_stolen) in detections:
        color = (0, 0, 255) if is_stolen else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

# Decode base64 image to OpenCV
def decode_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

# Webcam component
def browser_webcam_component():
    st.title("📷 Live Webcam Feed")
    webcam_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <style>
          video {
            width: 640px;
            height: 480px;
            border: 2px solid black;
          }
          button {
            margin-top: 10px;
            padding: 10px 20px;
            font-size: 16px;
          }
        </style>
      </head>
      <body>
        <h3>Webcam Feed</h3>
        <video id="webcam" autoplay></video>
        <br/>
        <button id="capture">Capture Frame</button>
        <script>
          const webcam = document.getElementById('webcam');
          const captureBtn = document.getElementById('capture');
          const canvas = document.createElement('canvas');
          canvas.style.display = 'none';
          document.body.appendChild(canvas);

          navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => { webcam.srcObject = stream; })
            .catch(error => { console.error("Error accessing webcam:", error); });

          captureBtn.addEventListener('click', () => {
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
            const frameData = canvas.toDataURL('image/jpeg');
            window.parent.location.hash = frameData;
          });
        </script>
      </body>
    </html>
    """
    components.html(webcam_html, height=600)

# Run app
def main():
    st.set_page_config(page_title="Smart Number Plate Detection", layout="wide")

    browser_webcam_component()

    st.subheader("📸 Captured Frame Detection")

    # Get base64 from URL hash
    captured_data = st.experimental_get_query_params().get('hash', [None])[0]

    if captured_data:
        base64_data = captured_data.replace("data:image/jpeg;base64,", "")
        frame = decode_base64_to_image(base64_data)
        detections = detect_number_plate(frame, conf_threshold=0.5)
        result_frame = draw_detections(frame, detections)

        for _, _, _, _, plate_text, is_stolen in detections:
            if is_stolen:
                st.error(f"🚨 ALERT: {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}")

        st.image(result_frame, channels="BGR", caption="Detected Frame from Webcam")

if __name__ == "__main__":
    main()
