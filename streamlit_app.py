
import streamlit as st
import cv2
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from detector.face_detector import detect_face
from detector.face_landmarks import detect_landmarks

st.set_page_config(page_title="EmotionVision Test", page_icon="😊")
st.title("EmotionVision - Detector and Webcam Test")

st.success("OpenCV, AV, WebRTC and detector imports loaded.")

class DetectorTestProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            face, bbox = detect_face(pil_image)

            if face is not None and bbox is not None:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                landmarks = detect_landmarks(pil_image)
                if landmarks:
                    for point in landmarks:
                        px = int(point["x"] * img.shape[1])
                        py = int(point["y"] * img.shape[0])
                        cv2.circle(img, (px, py), 1, (0, 255, 255), -1)
            else:
                cv2.putText(
                    img, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

        except Exception as e:
            print(f"Detector error: {type(e).__name__}: {e}")
            cv2.putText(
                img, "Detector error - check logs", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="detector-only-test",
    video_processor_factory=DetectorTestProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
