import streamlit as st
import cv2
import time
import tempfile
import os

from yolo_Backend import detect_frame

st.set_page_config(page_title="PPE Detection", layout="wide")

st.markdown("## 🦺 PPE Detection System")
st.markdown("Helmet & Safety Vest Detection (with violation screenshots)")

st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
skip_rate = st.sidebar.slider("Process every Nth frame", 1, 10, 2)
mode = st.sidebar.radio("Select Mode", ["Upload Video", "Webcam Live"])

# Folder for violations
os.makedirs("violations", exist_ok=True)

# ---------------------------
# VIDEO UPLOAD MODE
# ---------------------------
if mode == "Upload Video":
    uploaded = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info("Processing video... Please wait.")
        frame_win = st.empty()
        progress = st.progress(0)

        frame_no = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_no += 1

            # Skip frames for speed
            if frame_no % skip_rate != 0:
                continue

            annotated = detect_frame(frame, confidence)
            frame_win.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            progress.progress(min(frame_no / max(total_frames, 1), 1.0))

        cap.release()
        st.success("Video Completed ✔")

# ---------------------------
# WEBCAM MODE
# ---------------------------
elif mode == "Webcam Live":
    st.warning("Click Start to enable webcam")
    start = st.button("Start Camera")

    if start:
        stop_flag = st.checkbox("Stop Camera")
        cap = cv2.VideoCapture(0)
        frame_box = st.empty()

        while True:
            if stop_flag:
                st.info("Camera stopped by user.")
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Camera not detected!")
                break

            annotated = detect_frame(frame, confidence)
            frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()

