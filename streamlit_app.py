
import streamlit as st

st.set_page_config(
    page_title="EmotionVision Diagnostic",
    page_icon="😊"
)

st.title("EmotionVision - MediaPipe Test")

try:
    st.write("Step 1: Importing OpenCV...")
    import cv2
    st.success(f"OpenCV imported: {cv2.__version__}")

    st.write("Step 2: Importing MediaPipe...")
    import mediapipe as mp
    st.success(f"MediaPipe imported: {mp.__version__}")

    st.write("Step 3: Importing face detector...")
    from detector.face_detector import detect_face
    st.success("Face detector imported successfully.")

    st.write("Step 4: Importing face landmarks...")
    from detector.face_landmarks import detect_landmarks
    st.success("Face landmarks imported successfully.")

except Exception as e:
    st.exception(e)
