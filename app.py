import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import hashlib
import numpy as np

# Dummy stolen plate data
encrypted_stolen_plates = {
    hashlib.sha256("TN01AB1234".encode()).hexdigest(): "Coimbatore",
    hashlib.sha256("KL07CD5678".encode()).hexdigest(): "Chennai"
}

# Number Plate Detection Simulation
def detect_number_plate(frame):
    h, w, _ = frame.shape
    # Dummy one detection
    return [(int(w/4), int(h/4), int(w/2), int(h/2), "TN01AB1234", True)]

# Draw detections
def draw_detections(frame, detections):
    for (x1, y1, x2, y2, plate_text, is_stolen) in detections:
        color = (0, 0, 255) if is_stolen else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

# Custom Video Transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_to_display = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        detections = detect_number_plate(img)
        result_img = draw_detections(img, detections)

        for _, _, _, _, plate_text, is_stolen in detections:
            if is_stolen:
                st.session_state['alert'] = f"🚨 ALERT: {plate_text} - {encrypted_stolen_plates[hashlib.sha256(plate_text.encode()).hexdigest()]}"
            else:
                st.session_state['alert'] = None

        return result_img

# App Main
def main():
    st.title("🚘 Smart Number Plate Detection via Webcam")

    if 'alert' not in st.session_state:
        st.session_state['alert'] = None

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if st.session_state['alert']:
        st.error(st.session_state['alert'])

if __name__ == "__main__":
    main()
